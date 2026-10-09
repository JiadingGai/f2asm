"""Family-neutral artifact authentication; no compiler, assembler, or CUDA imports."""
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checked_files(directory, hashes):
    root = Path(directory).resolve()
    for name, expected in hashes.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError(f"Unsafe artifact path: {name}")
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"Artifact provenance mismatch: {name}")


def authenticated_candidate(directory, variant, source_path):
    directory = Path(directory)
    report = json.loads((directory / "qualification.json").read_text())
    if (report["id"] != variant.id or report["parameters"] != variant.parameters
            or report["status"] != "sass_encoded" or report.get("static_errors")):
        raise ValueError("Variant/qualification mismatch or static failure")
    binary = (directory / "candidate.cubin").read_bytes()
    source = Path(source_path).read_bytes()
    if sha256(binary) != report["candidate_sha256"] or sha256(source) != report["source_sha256"]:
        raise ValueError("Candidate/SASS changed after qualification")
    archived_source = directory.parent / "sass" / (variant.id + ".sass")
    if archived_source.read_bytes() != source:
        raise ValueError("Qualified SASS differs from the selected source")
    if sha256((directory / "candidate.nvdisasm.json").read_bytes()) != report["disassembly_sha256"]:
        raise ValueError("Disassembly changed")
    return binary, report
