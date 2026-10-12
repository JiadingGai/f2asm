"""Family-neutral artifact authentication; no compiler, assembler, or CUDA imports."""
import hashlib
import json
import math
from pathlib import Path

from .metrics import rates


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


def validate_result(root, manifest, job, attempt, verify_hash=False):
    """Authenticate a completed campaign attempt before reuse or reporting."""
    root = Path(root).resolve()
    relative = Path(attempt["path"])
    directory = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not directory.is_relative_to(root):
        raise ValueError("Unsafe attempt path")
    path = directory / "run.json"
    if path.is_symlink():
        raise ValueError("Symlinked result is not valid campaign evidence")
    blob = path.read_bytes()
    if verify_hash and sha256(blob) != attempt.get("run_sha256"):
        raise ValueError("Completed result hash mismatch")
    report = json.loads(blob)
    proof = report.get("correctness")
    if (report.get("variant") != job["variant"] or report.get("launch") != job["launch"]
            or report.get("candidate_sha256") != job.get("candidate_sha256")
            or report.get("campaign_status") != "requested_batches_completed"
            or not isinstance(proof, dict) or proof.get("passed") is not True):
        raise ValueError("Incomplete, failed or mismatched worker result")
    if proof.get("launch") != dict(job["launch"], repetitions=min(2, job["launch"]["repetitions"])):
        raise ValueError("Mismatched smoke/proof launch")
    for key in ("name", "ordinal", "driver_version", "pci_bus_id", "compute_capability", "sm_count", "l2_bytes"):
        if report.get("device", {}).get(key) != manifest["device"].get(key):
            raise ValueError("Worker device differs from preflight")
    expected_sources = {name: digest for name, digest in manifest["sources"].items()
                        if len(Path(name).parts) == 2 and name.endswith(".py")
                        and Path(name).parts[0] in ("common", job["family"])}
    if report.get("host_source_hashes") != expected_sources:
        raise ValueError("Worker host source differs from campaign preflight")
    checked_files(directory / "host-source-snapshot", expected_sources)
    raw = report.get("raw_samples", [])
    if not isinstance(raw, list) or len(raw) != job["samples"] or not raw:
        raise ValueError("Partial or extra sample count")

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    for sample in raw:
        cycles = sample.get("cycles", [])
        if (sample.get("numerical_check") != "passed" or not positive(sample.get("event_ms"))
                or len(cycles) != job["launch"]["blocks"] or not all(map(positive, cycles))
                or sample.get("cycles_per_iteration") != [c / job["launch"]["repetitions"] for c in cycles]):
            raise ValueError("Failed or invalid timing sample")
        if sample.get("work") != job["work"] or sample.get("rates") != rates(job["work"], sample["event_ms"]):
            raise ValueError("Timing workload/rate mismatch")
    if sha256((directory / "first-timing-output.bin").read_bytes()) != raw[0].get("output_sha256"):
        raise ValueError("Retained timing output mismatch")
    if sha256((directory / "correctness-output.bin").read_bytes()) != report.get("correctness_output_sha256"):
        raise ValueError("Retained smoke/proof output mismatch")
    return sha256(blob)
