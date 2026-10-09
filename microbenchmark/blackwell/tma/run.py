"""TMA-only host runner. Dry-run by default; no runtime assembly or training."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from ..common.artifacts import authenticated_candidate, sha256
from ..common.metrics import rates, summarize
from ..common.telemetry import telemetry
from .catalog import variants
from .inputs import Launch, check, prepare, validate_launch

SASS = Path(__file__).resolve().parent / "sass"


def work(variant, launch):
    return {"transferred_bytes": launch.blocks * launch.repetitions * variant.parameters["transfer_bytes"]}


def execute(args, variant, launch, directory, driver_factory=None):
    binary, qualification = authenticated_candidate(
        args.qualified / variant.id, variant, SASS / (variant.id + ".sass"))
    data, sizes = prepare(variant, launch)
    if driver_factory is None:
        from ..common.cuda_driver import Driver
        driver_factory = Driver
    driver = driver_factory(args.device)
    report = {"variant": variant.to_dict(), "launch": asdict(launch), "device": driver.info,
              "python_version": sys.version, "candidate_sha256": sha256(binary),
              "input_sha256": sha256(data), "raw_samples": [], "unknowns": qualification["unknowns"],
              "measurement_class": "experimental_not_publication_qualified",
              "cycle_metric": "clock_delta_including_loop_and_completion_overhead",
              "event_metric": "whole_kernel_including_setup_and_correctness_export",
              "traffic_metric": "requested transfer bytes, not distinct HBM traffic",
              "performance_claim_eligible": False}
    try:
        snapshot = directory / "host-source-snapshot"
        report["host_source_hashes"] = {}
        root = Path(__file__).resolve().parents[1]
        for component in ("tma", "common"):
            for source in (root / component).glob("*.py"):
                relative = str(source.relative_to(root))
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                blob = source.read_bytes()
                path.write_bytes(blob)
                report["host_source_hashes"][relative] = sha256(blob)
        inp = driver.upload(data)
        out = driver.upload(bytes([0xcd]) * sizes["output_bytes"])
        p = variant.parameters
        tensor = driver.tensor_map(inp, p["shape"], launch.count, p["element_bytes"]) if p["shape"] else 0
        function = driver.load(binary)
        proof = replace(launch, repetitions=min(2, launch.repetitions))
        driver.launch(function, proof.blocks, sizes["threads"], inp, out,
                      proof.repetitions, proof.count, tensor)
        driver.call("cuCtxSynchronize")
        raw = driver.download(out, sizes["output_bytes"])
        (directory / "correctness-output.bin").write_bytes(raw)
        report["correctness"] = check(variant, proof, raw, data)
        report["correctness"]["launch"] = asdict(proof)
        started = time.monotonic()
        workload = work(variant, launch)
        device = driver.info["pci_bus_id"]
        for sample_index in range(args.samples):
            if time.monotonic() - started > args.max_seconds:
                report["campaign_status"] = "duration_cap_reached"
                break
            before = telemetry(device)
            elapsed = driver.timed(lambda: driver.launch(
                function, launch.blocks, sizes["threads"], inp, out,
                launch.repetitions, launch.count, tensor))
            raw = driver.download(out, sizes["output_bytes"])
            checked = check(variant, launch, raw, data)
            sample = {"event_ms": elapsed, "launches": 1, "work": workload,
                      "rates": rates(workload, elapsed), "cycles": checked["cycles"],
                      "numerical_check": "passed", "output_sha256": sha256(raw),
                      "telemetry_before": before, "telemetry_after": telemetry(device)}
            if sample_index == 0:
                (directory / "first-timing-output.bin").write_bytes(raw)
            report["raw_samples"].append(sample)
            (directory / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            report["campaign_status"] = "requested_batches_completed"
        if report["raw_samples"]:
            report["event_ms_statistics"] = summarize([s["event_ms"] for s in report["raw_samples"]])
    except Exception as exc:
        report["campaign_status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        driver.close()
        (directory / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=[v.id for v in variants()])
    parser.add_argument("--qualified", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--memory-limit", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-unvalidated-hardware-protocols", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    variant = next(v for v in variants() if v.id == args.variant)
    launch = Launch(args.blocks, args.repetitions, args.count, args.seed, args.memory_limit)
    sizes = validate_launch(variant, launch)
    if args.samples <= 0 or not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        parser.error("Invalid campaign limits")
    if not args.execute or args.dry_run:
        print(json.dumps({"variant": variant.to_dict(), "launch": asdict(launch), "sizes": sizes,
                          "work": work(variant, launch), "hardware_execution": "NOT_RUN"}, indent=2))
        return
    if not args.accept_unvalidated_hardware_protocols or not args.output or not args.qualified:
        parser.error("Execution requires --qualified, --output, and --accept-unvalidated-hardware-protocols")
    if args.worker:
        execute(args, variant, launch, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "-m", "microbenchmark.blackwell.tma.run", *sys.argv[1:], "--worker"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=args.max_seconds + 60)
        (args.output / "stdout.txt").write_text(result.stdout)
        (args.output / "stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"GPU worker failed; see {args.output / 'stderr.txt'}")
        print(f"Experimental measurements saved to {args.output / 'run.json'}")
    except subprocess.TimeoutExpired as exc:
        (args.output / "timeout.json").write_text(json.dumps(
            {"status": "host_watchdog_timeout", "gpu_recovery": "UNKNOWN"}) + "\n")
        raise RuntimeError("Host timed out; inspect/reset the isolated GPU before retrying") from exc


if __name__ == "__main__":
    main()
