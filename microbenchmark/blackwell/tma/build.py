"""Offline TMA seed compilation and explicit-SASS F2Asm reconstruction, one worker."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

from ..common.artifacts import checked_files, sha256
from ..common.sass import audit, dot, normalized, parse_source, records
from .catalog import variants
from .protocols import inspect
from .seeds import generate

ROOT = Path(__file__).resolve().parents[1]
TMA = Path(__file__).resolve().parent


def selected(identifier=None):
    return [v for v in variants() if identifier is None or v.id == identifier]


def snapshot_sources():
    paths = sorted((ROOT / "common").glob("*.py")) + sorted(TMA.glob("*.py"))
    paths += [TMA / "manifest.json"] + sorted((TMA / "sass").glob("*.sass"))
    return {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}


def write_snapshot(directory, sources):
    for name, data in sources.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {name: sha256(data) for name, data in sources.items()}


def compile_shells(args):
    baseline = {r["id"]: r for r in json.loads((TMA / "manifest.json").read_text())["variants"]}
    sources = snapshot_sources()
    tools = {}
    for name in ("ptxas", "nvdisasm"):
        path = getattr(args, name).resolve()
        tools[name] = {"path": str(path), "sha256": sha256(path.read_bytes()),
                       "version": subprocess.check_output([str(path), "--version"],
                                                          text=True, timeout=30)}
    if "V12.8.93" not in tools["ptxas"]["version"] or "V13.4.92" not in tools["nvdisasm"]["version"]:
        raise ValueError("Pinned baseline requires ptxas 12.8.93 and nvdisasm 13.4.92")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = write_snapshot(args.output / "sources", sources)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAX_JOBS="1")
    results = []
    for variant in selected(args.variant):
        source = generate(variant).encode()
        if sha256(source) != baseline[variant.id]["ptx_sha256"]:
            raise ValueError(f"PTX changed from pinned baseline: {variant.id}")
        out = args.output / variant.id
        out.mkdir()
        (out / "seed.ptx").write_bytes(source)
        row = {**variant.to_dict(), "source_sha256": sha256(source),
               "commands": [], "status": "compiler_failed"}
        commands = [
            ("compile", [tools["ptxas"]["path"], "-arch=sm_100a", "-O3", "-v", "--preserve-relocs",
                         str(out / "seed.ptx"), "-o", str(out / "seed.cubin")]),
            ("disassembly", [tools["nvdisasm"]["path"], "-json", str(out / "seed.cubin")]),
            ("sass", [tools["nvdisasm"]["path"], "-c", "-hex", str(out / "seed.cubin")]),
        ]
        for name, argv in commands:
            try:
                proc = subprocess.run(argv, capture_output=True, timeout=120, env=env)
            except subprocess.TimeoutExpired:
                row["status"] = "build_timeout"
                break
            (out / (name + ".stdout")).write_bytes(proc.stdout)
            (out / (name + ".stderr")).write_bytes(proc.stderr)
            row["commands"].append({"argv": argv, "returncode": proc.returncode})
            if proc.returncode:
                row["status"] = "compiler_failed" if name == "compile" else "inspection_failed"
                break
        else:
            row["status"] = "shell_built"
        row["files"] = {p.name: sha256(p.read_bytes()) for p in out.iterdir() if p.is_file()}
        row["hardware_correctness"] = "NOT_RUN"
        results.append(row)
        print(f"{row['status']}: {variant.id}", flush=True)
    for name, tool in tools.items():
        if sha256(Path(tool["path"]).read_bytes()) != tool["sha256"]:
            raise ValueError(f"Tool changed during build: {name}")
    if snapshot_sources() != sources:
        raise ValueError("Source changed during build")
    report = {"schema": "blackwell-tma-build-v1", "target": "sm_100a", "workers": 1,
              "tools": tools, "source_hashes": source_hashes, "results": results,
              "counts": dict(Counter(r["status"] for r in results)), "hardware_correctness": "NOT_RUN"}
    (args.output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(row["status"] != "shell_built" for row in results):
        raise RuntimeError("Incomplete build; see build-report.json")


def assemble_words(rows, base, supplement=None):
    from f2asm.bitlinear import BitLinearError
    words, supplemental = [], 0
    for row in rows:
        try:
            encoded = base.assemble(row["sass"], address=row["pc"])
        except BitLinearError:
            if supplement is None:
                raise
            encoded = supplement.assemble(row["sass"], address=row["pc"])
            supplemental += 1
        if encoded.output_mask != base.profile.output_mask:
            raise ValueError("Partial bit ownership")
        words.append(encoded.compose(control=row["control"]))
    return words, supplemental


def assemble(args):
    # Import F2Asm only for offline assembly, never for listing/dry-run/execution.
    import f2asm
    from f2asm.bitlinear_repository import BitLinearRepository, MAX_REPOSITORY_JSON_BYTES
    from f2asm.cubin import Cubin
    from f2asm.encoder import Encoder

    def load_model(path):
        with path.open("rb") as handle:
            data = handle.read(MAX_REPOSITORY_JSON_BYTES + 1)
        if len(data) > MAX_REPOSITORY_JSON_BYTES:
            raise ValueError("Model too large")
        return Encoder(BitLinearRepository.from_json(data.decode()), target="sm_100a"), sha256(data)

    base, base_hash = load_model(args.model)
    supplement, supplement_hash = load_model(args.supplement) if args.supplement else (None, None)
    sources = snapshot_sources()
    baseline = {r["id"]: r for r in json.loads((TMA / "manifest.json").read_text())["variants"]}
    build_bytes = (args.build / "build-report.json").read_bytes()
    build = json.loads(build_bytes)
    checked_files(args.build / "sources", build["source_hashes"])
    entries = {row["id"]: row for row in build["results"]}
    if len(entries) != len(build["results"]):
        raise ValueError("Duplicate build entries")
    chosen = selected(args.variant)
    if any(v.id not in entries for v in chosen):
        raise ValueError("Missing TMA build entries; supply a complete build or --variant")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = write_snapshot(args.output / "sources", sources)
    f2asm_sources = {p.name: p.read_bytes() for p in Path(f2asm.__file__).parent.glob("*.py")}
    f2asm_hashes = write_snapshot(args.output / "f2asm-source-snapshot", f2asm_sources)
    sass_dir = args.output / "sass"
    sass_dir.mkdir()
    results = []
    for variant in chosen:
        row, pinned = entries[variant.id], baseline[variant.id]
        if row["status"] != "shell_built" or row["parameters"] != variant.parameters:
            raise ValueError(f"Failed/mismatched build: {variant.id}")
        directory = args.build / variant.id
        checked_files(directory, row["files"])
        raw = (directory / "seed.cubin").read_bytes()
        ptx = (directory / "seed.ptx").read_bytes()
        source = sources["tma/sass/" + variant.id + ".sass"]
        if (sha256(ptx) != pinned["ptx_sha256"] or sha256(source) != pinned["source_sha256"]
                or sha256(raw) != pinned["candidate_sha256"]):
            raise ValueError(f"Changed source/compiler baseline requires review: {variant.id}")
        cubin = Cubin(raw)
        if cubin.flags != 0x0600640a or len(cubin.executable_sections) != 1:
            raise ValueError("Wrong target/layout")
        section = cubin.executable_sections[0]
        if section.name != ".text.bench":
            raise ValueError("Wrong kernel")
        words = [int.from_bytes(word, "little") for word in cubin.iter_instructions(section)]
        disasm = (directory / "disassembly.stdout").read_bytes()
        reference = records(disasm.decode(), words, (directory / "sass.stdout").read_text())
        candidate = parse_source(source.decode())
        if len(candidate) != len(reference) or any(
                normalized(a["sass"]) != normalized(b["sass"]) or a["control"] != b["control"]
                for a, b in zip(candidate, reference)):
            raise ValueError("Changed SASS needs a reviewed fixed-layout patch contract")
        encoded, supplemental = assemble_words(candidate, base, supplement)
        if encoded != words:
            raise ValueError("Full instruction/control round-trip mismatch")
        binary = cubin.replace_sections({section.index: b"".join(w.to_bytes(16, "little") for w in encoded)})
        if binary != raw:
            raise ValueError("Non-executable bytes changed")
        analysis = audit(candidate, "tma")
        analysis["protocol"] = inspect(variant, ptx.decode(), candidate)
        analysis["errors"].extend(analysis["protocol"]["errors"])
        if analysis["errors"]:
            raise ValueError(f"Static check failed: {analysis['errors']}")
        out = args.output / variant.id
        out.mkdir()
        (sass_dir / (variant.id + ".sass")).write_bytes(source)
        (out / "candidate.cubin").write_bytes(binary)
        (out / "candidate.nvdisasm.json").write_bytes(disasm)
        (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        (out / "cfg.dot").write_text(dot(analysis["cfg"]))
        result = {**variant.to_dict(), "status": "sass_encoded", "instruction_words": len(encoded),
                  "base_model_words": len(encoded) - supplemental, "compiler_evidence_words": supplemental,
                  "candidate_sha256": sha256(binary), "source_sha256": sha256(source),
                  "disassembly_sha256": sha256(disasm), "metadata_unchanged": True,
                  "unknowns": analysis["unknowns"], "static_errors": [],
                  "hardware_correctness": "NOT_RUN", "performance": "NOT_MEASURED",
                  "gpu_launch_eligibility": "NEEDS_HARDWARE_QUALIFICATION"}
        (out / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(f"sass_encoded: {variant.id}", flush=True)
    if snapshot_sources() != sources:
        raise ValueError("Benchmark source changed during assembly")
    checked_files(Path(f2asm.__file__).parent, f2asm_hashes)
    report = {"schema": "blackwell-tma-qualification-v1", "target": "sm_100a", "workers": 1,
              "results": results, "model_sha256": base_hash, "supplement_sha256": supplement_hash,
              "build_report_sha256": sha256(build_bytes), "source_hashes": source_hashes,
              "f2asm_source_hashes": f2asm_hashes, "hardware_correctness": "NOT_RUN",
              "performance": "NOT_MEASURED", "training_performed": False}
    (args.output / "qualification-report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compiler = commands.add_parser("compile", help="Compile pinned GPU-free PTX seeds sequentially")
    compiler.add_argument("--ptxas", required=True, type=Path)
    compiler.add_argument("--nvdisasm", required=True, type=Path)
    assembler = commands.add_parser("assemble", help="Encode checked-in SASS using existing F2Asm models")
    assembler.add_argument("--build", required=True, type=Path)
    assembler.add_argument("--model", required=True, type=Path)
    assembler.add_argument("--supplement", type=Path, help="Explicit existing compiler-evidence model; never trained implicitly")
    for subparser in (compiler, assembler):
        subparser.add_argument("--output", required=True, type=Path)
        subparser.add_argument("--variant", choices=[v.id for v in variants()])
    args = parser.parse_args()
    (compile_shells if args.command == "compile" else assemble)(args)


if __name__ == "__main__":
    main()
