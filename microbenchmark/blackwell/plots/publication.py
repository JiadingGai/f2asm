"""Export de-identified measurements; verify/replot without the private raw archive."""
import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path

from ..campaign import validate_completed
from ..common.artifacts import checked_files, sha256
from ..common.metrics import summarize
from ..common.reporting import _collect, _csv, DISCLAIMER, LATENCY_NOTE, THROUGHPUT_NOTE

DEVICE_FIELDS = ("name", "compute_capability", "sm_count", "l2_bytes", "driver_version")
DATA_FILES = {"summary.json", "summary.csv", "samples.csv.gz"}


def export(results, output):
    manifest = json.loads((results / "manifest.json").read_text())
    state = json.loads((results / "state.json").read_text())
    counts = validate_completed(results, manifest, state)
    rows, samples = _collect(results, manifest, state)
    device = {key: manifest["device"][key] for key in DEVICE_FIELDS}
    summary = {"schema": "blackwell-campaign-summary-v1", "campaign_status": "completed",
               "device": device, "status_counts": {"passed": len(rows)},
               "measurement_class": "experimental_not_publication_qualified",
               "performance_claim_eligible": False, "notes": [DISCLAIMER, LATENCY_NOTE, THROUGHPUT_NOTE],
               "configurations": rows}
    blobs = {"summary.json": (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode(),
             "summary.csv": _csv(rows, ("job_id", "family", "kind", "variant", "status", "state_status")).encode(),
             "samples.csv.gz": gzip.compress(_csv(samples, ("job_id", "status", "sample_index", "numerical_check")).encode(), mtime=0)}
    provenance = {"schema": "blackwell-publication-v1", "campaign_identity": manifest["identity"],
                  "device": device, "runtime": manifest["runtime"], "options": manifest["options"],
                  "sources": manifest["sources"], "artifacts": manifest["artifacts"],
                  "counts": counts, "performance_claim_eligible": False,
                  "exporter_sha256": sha256(Path(__file__).read_bytes()),
                  "source_hashes": {name: sha256((results / name).read_bytes()) for name in ("manifest.json", "state.json")},
                  "run_hashes": {key: entry["attempts"][-1]["run_sha256"] for key, entry in state["jobs"].items()},
                  "file_hashes": {name: sha256(blob) for name, blob in blobs.items()},
                  "scope": "Derived measurements verified against the full archive at export; not a raw-evidence archive. "
                           "Host/GPU identifiers, raw output buffers and source snapshots are not included."}
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        (output / name).write_bytes(blob)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    load(output)
    return counts


def load(root):
    """Verify checksums and recompute summary statistics from individual samples.

    This checks the public derivative, not omitted raw GPU output buffers.
    """
    provenance = json.loads((root / "provenance.json").read_text())
    if provenance.get("schema") != "blackwell-publication-v1" or set(provenance["file_hashes"]) != DATA_FILES:
        raise ValueError("Unknown publication format")
    checked_files(root, provenance["file_hashes"])
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["configurations"]
    if (summary["campaign_status"] != "completed" or summary["device"] != provenance["device"]
            or summary.get("performance_claim_eligible") is not False or not rows
            or summary["status_counts"] != {"passed": len(rows)}):
        raise ValueError("Invalid published summary")
    if (root / "summary.csv").read_bytes() != _csv(rows, ("job_id", "family", "kind", "variant", "status", "state_status")).encode():
        raise ValueError("Summary CSV/JSON mismatch")
    samples = list(csv.DictReader(io.StringIO(gzip.decompress((root / "samples.csv.gz").read_bytes()).decode())))
    by_job = {row["job_id"]: [] for row in rows}
    if len(by_job) != len(rows):
        raise ValueError("Duplicate configuration")
    for sample in samples:
        if sample["job_id"] not in by_job or sample["status"] != "passed" or sample["numerical_check"] != "passed":
            raise ValueError("Unknown or failed published sample")
        by_job[sample["job_id"]].append(sample)
    for row in rows:
        group = by_job[row["job_id"]]
        if (row["status"] != "passed" or row["recorded_samples"] != row["requested_samples"]
                or len(group) != row["recorded_samples"] or not group
                or sorted(int(s["sample_index"]) for s in group) != list(range(len(group)))):
            raise ValueError("Incomplete/duplicate published samples")
        events = [float(s["event_ms"]) for s in group]
        cycles = []
        for sample in group:
            raw, normalized = json.loads(sample["cycles"]), json.loads(sample["cycles_per_iteration"])
            if (len(raw) != row["blocks"] or normalized != [v / row["repetitions"] for v in raw]
                    or any(not math.isfinite(v) or v <= 0 for v in raw)):
                raise ValueError("Invalid published cycles")
            cycles.extend(normalized)
        if any(not math.isfinite(v) or v <= 0 for v in events):
            raise ValueError("Invalid published event time")
        metrics = {"event_ms": events, "cycles_per_iteration": cycles}
        if row["kind"] == "throughput":
            volume = row["blocks"] * row["repetitions"] * row["transfer_bytes"]
            metrics["requested_gbps"] = [volume / value / 1e6 for value in events]
        for metric, values in metrics.items():
            # statistics implementations differ by a few ULPs between Python releases.
            if any(not math.isclose(row[metric + "_" + key], value, rel_tol=1e-12, abs_tol=0)
                   for key, value in summarize(values).items()):
                raise ValueError("Published summary differs from individual samples")
    if provenance["counts"] != {"passed_jobs": len(rows), "passed_samples": len(samples)}:
        raise ValueError("Published counts disagree")
    return provenance, rows, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Export raw campaign to a new public directory; omit to verify an export")
    args = parser.parse_args()
    counts = export(args.results, args.output) if args.output else load(args.results)[0]["counts"]
    print(json.dumps(counts))
