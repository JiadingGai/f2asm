"""Sequential Blackwell campaigns. No GPU access unless --execute is explicit."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from .common.artifacts import authenticated_candidate, sha256, validate_result
from .common.types import Launch

ROOT = Path(__file__).resolve().parent
SCHEMA = "blackwell-campaign-v2"


def families(suite):
    """Suite registry: later tensor/memory campaigns can supply other families."""
    if suite != "tma":
        raise ValueError(f"Unknown campaign suite: {suite}")
    from .tma import run as tma_run, sweeps as tma_sweeps
    from .memory import run as memory_run, sweeps as memory_sweeps
    return {"tma": (tma_run.suite(), tma_sweeps.plan),
            "memory": (memory_run.suite(), memory_sweeps.plan)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_hashes(suite):
    paths = list(ROOT.glob("*.py"))
    for component in ("common", *families(suite)):
        for pattern in ("*.py", "manifest.json", "sass/*.sass", "seeds/*.ptx"):
            paths.extend((ROOT / component).glob(pattern))
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in sorted(paths)}


def plan_jobs(suite, device, options):
    sm, l2 = device["sm_count"], device["l2_bytes"]
    if type(sm) is not int or type(l2) is not int or sm <= 0 or l2 <= 0:
        raise ValueError("Positive integer device SM/L2 geometry is required")
    jobs = []
    for family, (hooks, planner) in families(suite).items():
        for variant in hooks.variants():
            for row in planner(variant, sm, l2_bytes=l2, global_bytes=options["global_bytes"], seed=options["seed"]):
                launch = Launch(**row["launch"])
                key = sha256(canonical(asdict(launch)))[:12]
                jobs.append({"id": f"{family}-{variant.id}-{key}", "family": family,
                             "variant": variant.to_dict(), "launch": asdict(launch),
                             "memory": row["memory"], "work": hooks.work(variant, launch),
                             "kind": "latency" if family == "memory" or variant.parameters.get("fine") else "throughput",
                             "samples": options["samples"]})
    return jobs


def authenticate(suite, qualified):
    if set(qualified) != set(families(suite)):
        raise ValueError("Supply one --qualified FAMILY=PATH for each suite family: tma, memory")
    hashes = {}
    for family, (hooks, _) in families(suite).items():
        pinned = {r["id"]: r for r in json.loads((hooks.directory / "manifest.json").read_text())["variants"]}
        for variant in hooks.variants():
            directory = qualified[family] / variant.id
            binary, report = authenticated_candidate(directory, variant, hooks.directory / "sass" / (variant.id + ".sass"))
            if sha256(binary) != pinned[variant.id]["candidate_sha256"]:
                raise ValueError(f"Candidate differs from pinned baseline: {variant.id}")
            hashes[f"{family}/{variant.id}"] = {
                "candidate_sha256": sha256(binary),
                "qualification_sha256": sha256((directory / "qualification.json").read_bytes()),
                "disassembly_sha256": report["disassembly_sha256"], "source_sha256": report["source_sha256"]}
    return hashes


def manifest_for(suite, device, options, artifacts=None):
    jobs = plan_jobs(suite, device, options)
    for job in jobs:
        if artifacts:
            job["candidate_sha256"] = artifacts[f"{job['family']}/{job['variant']['id']}"]["candidate_sha256"]
    manifest = {"schema": SCHEMA, "suite": suite, "target": "B200/sm_100a", "device": device,
                "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
                "options": options, "sources": source_hashes(suite), "artifacts": artifacts or {}, "jobs": jobs,
                "measurement_class": "experimental_not_publication_qualified",
                "performance_claim_eligible": False,
                "protocol": "Original overlapping-stream, serial-per-CTA TMA baselines; ordinary-load reference has different node spacing/warmup"}
    manifest["identity"] = sha256(canonical(manifest))
    return manifest


def probe_device(ordinal):
    """Run only in the bounded preflight subprocess, never during offline planning."""
    from .common.cuda_driver import Driver
    driver = Driver(ordinal)
    try:
        info = dict(driver.info)
        result = subprocess.run(["nvidia-smi", f"--id={info['pci_bus_id']}",
                                 "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10)
        uuid, free, total = (v.strip() for v in result.stdout.strip().split(","))
        if not uuid.startswith("GPU-"):
            raise ValueError("A physical GPU UUID is required for safe campaign resume")
        info.update(gpu_uuid=uuid, host=platform.node())
        return {"device": info, "free_bytes": int(float(free) * 1024**2),
                "total_bytes": int(float(total) * 1024**2)}
    finally:
        driver.close()


class WorkerFailure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def bounded_process(argv, timeout, stdout, stderr):
    """Own the actual worker and its process group, including interrupt cleanup."""
    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise WorkerFailure("timeout", "Worker watchdog expired; GPU recovery is unknown") from exc


def preflight(ordinal):
    import tempfile
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        code = bounded_process([sys.executable, "-m", "microbenchmark.blackwell.campaign",
                                "--probe-device", str(ordinal)], 30, out, err)
        out.seek(0); err.seek(0)
        if code:
            raise RuntimeError("B200 preflight failed: " + err.read().decode(errors="replace"))
        return json.loads(out.read())


def worker(job, directory, qualified, options, ordinal):
    launch = job["launch"]
    argv = [sys.executable, "-m", f"microbenchmark.blackwell.{job['family']}.run",
            "--variant", job["variant"]["id"], "--qualified", str(qualified[job["family"]]),
            "--output", str(directory), "--device", str(ordinal), "--samples", str(job["samples"]),
            "--max-seconds", str(options["max_seconds"]), "--execute", "--worker",
            "--accept-unvalidated-hardware-protocols"]
    for key in ("blocks", "count", "repetitions", "seed", "memory_limit"):
        argv.extend(["--" + key.replace("_", "-"), str(launch[key])])
    with (directory / "stdout.txt").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
        code = bounded_process(argv, options["worker_timeout"], out, err)
    if code:
        raise WorkerFailure("failed", f"GPU worker exited {code}; inspect this attempt's stderr.txt")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def fresh_state(manifest):
    return {"schema": SCHEMA, "identity": manifest["identity"], "status": "pending",
            "jobs": {j["id"]: {"status": "pending", "attempts": []} for j in manifest["jobs"]}}


def check_manifest(manifest):
    content = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest.get("schema") != SCHEMA or sha256(canonical(content)) != manifest.get("identity"):
        raise ValueError("Campaign manifest integrity mismatch")


def validate_resume(root, manifest, state):
    check_manifest(manifest)
    if state.get("identity") != manifest["identity"] or set(state.get("jobs", {})) != {j["id"] for j in manifest["jobs"]}:
        raise ValueError("State does not match the immutable campaign manifest")
    for job in manifest["jobs"]:
        entry = state["jobs"][job["id"]]
        if entry["status"] == "passed":
            if not entry["attempts"] or entry["attempts"][-1].get("status") != "passed":
                raise ValueError("Passed state lacks a passed attempt")
            validate_result(root, manifest, job, entry["attempts"][-1], verify_hash=True)
        elif entry["status"] not in ("pending", "running", "failed", "timeout", "interrupted"):
            raise ValueError("Unknown campaign job state")


def check_retry(state, retry_failed):
    if any(e["status"] not in ("pending", "passed") for e in state["jobs"].values()) and not retry_failed:
        raise ValueError("Inspect/recover the GPU, then use --resume --retry-failed for failed/interrupted work")


def validate_completed(root, manifest, state):
    """Strict offline verification, including retained outputs/source snapshots."""
    validate_resume(root, manifest, state)
    if (state.get("status") != "completed" or not manifest["jobs"]
            or any(entry["status"] != "passed" for entry in state["jobs"].values())):
        raise ValueError("Campaign is incomplete; retain the GPU/archive until recovery")
    return {"passed_jobs": len(manifest["jobs"]),
            "passed_samples": sum(job["samples"] for job in manifest["jobs"])}


def execute_campaign(root, manifest, state, qualified, *, run_job=worker, retry_failed=False):
    from .common.reporting import write_reports
    validate_resume(root, manifest, state)
    check_retry(state, retry_failed)
    for entry in state["jobs"].values():
        if entry["status"] == "running":
            entry["status"] = entry["attempts"][-1]["status"] = "interrupted"
    state["status"] = "running"
    try:
        for job in manifest["jobs"]:
            entry = state["jobs"][job["id"]]
            if entry["status"] == "passed":
                continue
            try:
                if source_hashes(manifest["suite"]) != manifest["sources"] or authenticate(manifest["suite"], qualified) != manifest["artifacts"]:
                    raise ValueError("Source/artifacts changed during campaign; use a new campaign")
            except Exception as exc:
                state.update(status="blocked", error=str(exc))
                raise
            relative = Path("runs") / job["id"] / f"attempt-{len(entry['attempts']) + 1:03d}"
            directory = root / relative
            attempt = {"path": relative.as_posix(), "status": "running", "started_unix": time.time()}
            entry["attempts"].append(attempt)
            entry["status"] = "running"
            # Reserve durably before mkdir/spawn; retries preserve every attempt.
            atomic_json(root / "state.json", state)
            print(f"{job['variant']['id']} blocks={job['launch']['blocks']} count={job['launch']['count']}", flush=True)
            try:
                directory.mkdir(parents=True, exist_ok=False)
                run_job(job, directory, qualified, manifest["options"], manifest["device"]["ordinal"])
                attempt["run_sha256"] = validate_result(root, manifest, job, attempt)
                attempt["status"] = entry["status"] = "passed"
                entry.pop("error", None)
            except (Exception, KeyboardInterrupt) as exc:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else getattr(exc, "status", "failed")
                attempt.update(status=status, error=str(exc) or "Keyboard interrupt")
                entry.update(status=status, error=attempt["error"])
                state.update(status="stopped", error=attempt["error"])
                raise
            finally:
                attempt["finished_unix"] = time.time()
                atomic_json(root / "state.json", state)
        state["status"] = "completed"
        state.pop("error", None)
    finally:
        atomic_json(root / "state.json", state)
        write_reports(root, manifest, state)


@contextmanager
def campaign_lock(root):
    with (root / ".campaign.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this campaign directory") from exc
        yield  # Closing the file releases the lock, including on exceptions.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["tma"], default="tma", help="TMA plus ordinary-load latency reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualified", action="append", default=[], metavar="FAMILY=PATH")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    modes.add_argument("--verify-only", action="store_true", help="Fail unless all raw campaign evidence verifies")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only after inspecting/recovering the GPU")
    parser.add_argument("--accept-unvalidated-hardware-protocols", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sm-count", type=int, help="Offline plan only; execution queries the GPU")
    parser.add_argument("--l2-bytes", type=int, help="Offline plan only; execution queries the GPU")
    parser.add_argument("--global-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-seconds", type=float, default=60, help="Per-configuration sampling duration cap")
    parser.add_argument("--worker-timeout", type=float, default=300, help="Wall-time watchdog including input/oracle setup and cleanup")
    parser.add_argument("--probe-device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_device is not None:
        print(json.dumps(probe_device(args.probe_device)))
        return
    from .common.reporting import write_reports
    if args.report_only or args.verify_only:
        if not args.output:
            parser.error("Offline reporting/verification requires --output")
        with campaign_lock(args.output):
            manifest = json.loads((args.output / "manifest.json").read_text())
            check_manifest(manifest)
            state = json.loads((args.output / "state.json").read_text())
            if state.get("identity") != manifest["identity"]:
                raise ValueError("Report state/manifest mismatch")
            if args.verify_only:
                print(json.dumps(validate_completed(args.output, manifest, state)))
            else:
                write_reports(args.output, manifest, state)
        return
    if (args.samples <= 0 or args.global_bytes <= 0 or not positive(args.max_seconds)
            or not positive(args.worker_timeout) or args.max_seconds > 3600
            or args.worker_timeout < args.max_seconds or args.worker_timeout > 86400 or args.device < 0):
        parser.error("Invalid campaign limits")
    options = {key: getattr(args, key) for key in ("samples", "seed", "global_bytes", "max_seconds", "worker_timeout")}
    qualified = {}
    for value in args.qualified:
        family, sep, path = value.partition("=")
        if not sep or not path or family in qualified:
            parser.error("Use --qualified FAMILY=PATH exactly once per family")
        qualified[family] = Path(path).resolve()
    if not args.execute:
        if args.resume or args.retry_failed:
            parser.error("Resume/retry applies only to --execute")
        if args.sm_count is None or args.l2_bytes is None:
            parser.error("Offline plans require --sm-count and --l2-bytes; no GPU is queried")
        device = {"name": "OFFLINE_GEOMETRY_NOT_GPU_EVIDENCE", "sm_count": args.sm_count, "l2_bytes": args.l2_bytes}
        manifest = manifest_for(args.suite, device, options)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            state = fresh_state(manifest)
            state["status"] = "dry_run"
            atomic_json(args.output / "manifest.json", manifest)
            atomic_json(args.output / "state.json", state)
            write_reports(args.output, manifest, state)
        print(json.dumps({"hardware_execution": "NOT_RUN", "measurement_configurations": len(manifest["jobs"]),
                          "manifest": manifest if not args.output else str(args.output / "manifest.json")}, indent=2))
        return
    if not args.output or not args.accept_unvalidated_hardware_protocols:
        parser.error("Execution requires --output and --accept-unvalidated-hardware-protocols")
    if args.sm_count is not None or args.l2_bytes is not None:
        parser.error("Execution uses detected GPU geometry, not offline overrides")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    artifacts = authenticate(args.suite, qualified)
    if args.resume:
        if not args.output.is_dir():
            parser.error("Resume directory does not exist")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with campaign_lock(args.output):
        if args.resume:
            prior = json.loads((args.output / "state.json").read_text())
            check_retry(prior, args.retry_failed)
        observed = preflight(args.device)
        manifest = manifest_for(args.suite, observed["device"], options, artifacts)
        required = max(j["memory"]["input_bytes"] + j["memory"]["output_bytes"] + 128 for j in manifest["jobs"])
        if observed["free_bytes"] < required + 256 * 1024**2:
            raise ValueError("Insufficient free GPU memory for the plan plus safety reserve")
        atomic_json(args.output / f"preflight-{time.time_ns()}.json", observed)
        if args.resume:
            saved = json.loads((args.output / "manifest.json").read_text())
            if saved != manifest:
                raise ValueError("Resume identity changed (source/artifact/device/options); start a fresh output directory")
            state = json.loads((args.output / "state.json").read_text())
        else:
            atomic_json(args.output / "manifest.json", manifest)
            state = fresh_state(manifest)
            atomic_json(args.output / "state.json", state)
        execute_campaign(args.output, manifest, state, qualified, retry_failed=args.retry_failed)


if __name__ == "__main__":
    # A normal process termination must also reap the active worker group.
    # SIGKILL cannot be caught; its running checkpoint requires recovery on resume.
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    main()
