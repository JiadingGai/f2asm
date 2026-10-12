"""CPU-only synthetic report tests; never treated as GPU measurements."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from ..reporting import write_reports
from ..metrics import rates


DEVICE = {"name": "Synthetic & offline", "ordinal": 0, "driver_version": 13000,
          "pci_bus_id": "0000:00:00.0", "compute_capability": [10, 0],
          "sm_count": 2, "l2_bytes": 1024}
SOURCE_BLOBS = {"common/runner.py": b"synthetic common source\n",
                "tma/run.py": b"synthetic TMA source\n", "memory/run.py": b"synthetic memory source\n"}
SOURCES = {name: sha256(blob).hexdigest() for name, blob in SOURCE_BLOBS.items()}


def job(identifier, *, kind="latency", family="tma", size=1024, blocks=1, case=None, shape=None):
    return {"id": identifier, "kind": kind, "family": family,
            "variant": {"id": identifier, "case": case or ("tma_completion_latency" if kind == "latency"
                                                            else "tma_bulk_throughput"),
                        "family": family, "parameters": {"transfer_bytes": 16 if kind == "latency" else size,
                                                          "shape": shape or []}},
            "launch": {"blocks": blocks, "repetitions": 8, "count": 64, "seed": 17, "memory_limit": 1048576},
            "memory": {"input_bytes": size, "output_bytes": 64, "threads": 32},
            "work": {"requested_bytes": blocks * 8 * (16 if kind == "latency" else size)},
            "samples": 2, "candidate_sha256": "candidate"}


def save_run(root, item, status="passed", **overrides):
    relative = Path("runs") / item["id"] / "attempt-001"
    directory = root / relative
    directory.mkdir(parents=True)
    sources = {name: digest for name, digest in SOURCES.items()
               if Path(name).parts[0] in ("common", item["family"])}
    for name in sources:
        path = directory / "host-source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCE_BLOBS[name])
    output = bytes(item["memory"]["output_bytes"])
    for name in ("correctness-output.bin", "first-timing-output.bin"):
        (directory / name).write_bytes(output)
    run = {"variant": item["variant"], "launch": item["launch"],
           "candidate_sha256": "candidate", "device": DEVICE, "host_source_hashes": sources,
           "correctness": {"passed": True, "launch": dict(item["launch"], repetitions=2)},
           "correctness_output_sha256": sha256(output).hexdigest(),
           "campaign_status": "requested_batches_completed",
           "raw_samples": [{"event_ms": 2 + index, "cycles": [80 + index * 16] * item["launch"]["blocks"],
                            "cycles_per_iteration": [10 + index * 2] * item["launch"]["blocks"],
                            "numerical_check": "passed", "cpu_check_seconds": 0.00001,
                            "work": item["work"], "rates": rates(item["work"], 2 + index),
                            "output_sha256": sha256(output).hexdigest(),
                            "telemetry_before": {"temperature_c": 40},
                            "telemetry_after": {"temperature_c": 41}} for index in range(item["samples"])]}
    run.update(overrides)
    blob = (json.dumps(run) + "\n").encode()
    (directory / "run.json").write_bytes(blob)
    return {"status": status, "attempts": [{"status": status, "path": str(relative),
                                             "run_sha256": sha256(blob).hexdigest()}]}


class ReportingTests(unittest.TestCase):
    def manifest(self, jobs):
        return {"schema": "test", "device": DEVICE, "sources": SOURCES, "jobs": jobs}

    def summary(self, root):
        return json.loads((root / "summary.json").read_text())

    def test_passed_missing_failed_and_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = job("good")
            failed = job("failed", size=2048)
            missing = job("missing", size=4096)
            pending = job("pending")
            state = {"status": "failed", "jobs": {
                "good": save_run(root, good), "failed": save_run(root, failed, "failed"),
                "missing": {"status": "passed", "attempts": [{"status": "passed", "path": "runs/missing",
                                                              "run_sha256": "absent"}]}}}
            write_reports(root, self.manifest([good, failed, missing, pending]), state)
            summary = self.summary(root)
            self.assertEqual(summary["status_counts"], {"failed": 1, "missing": 1, "passed": 1, "pending": 1})
            self.assertFalse(summary["performance_claim_eligible"])
            rows = {row["job_id"]: row for row in summary["configurations"]}
            self.assertEqual(rows["good"]["cycles_per_iteration_median"], 11)
            self.assertNotIn("cycles_per_iteration_median", rows["failed"])
            self.assertNotIn("cycles_per_iteration_median", rows["missing"])
            with (root / "summary.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            with (root / "samples.csv").open() as stream:
                raw = list(csv.DictReader(stream))
            self.assertEqual(len(raw), 4)
            self.assertEqual({row["status"] for row in raw}, {"passed", "failed"})
            for name in ("latency.svg", "throughput.svg"):
                ET.parse(root / name)
            self.assertIn("1/4 passed", (root / "latency.svg").read_text())
            self.assertIn("Warmup differs", (root / "latency.svg").read_text())

    def test_all_pending_plots_and_empty_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [job("latency"), job("ldg", family="memory"),
                     job("throughput", kind="throughput", blocks=2)]
            write_reports(root, self.manifest(items), {"status": "planned", "jobs": {}})
            self.assertEqual(self.summary(root)["status_counts"], {"pending": 3})
            for name in ("latency.svg", "throughput.svg"):
                text = (root / name).read_text()
                self.assertIn("No validated measurements yet", text)
                self.assertNotIn("<circle", text)
                ET.fromstring(text)
            with (root / "samples.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_authentication_and_configuration_fail_closed(self):
        mutations = ("hash", "missing_hash", "candidate", "launch", "partial", "bad_check", "bad_sample", "bad_cycles")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                item = job("test")
                overrides = {}
                if mutation == "candidate":
                    overrides["candidate_sha256"] = "wrong"
                if mutation == "launch":
                    overrides["launch"] = dict(item["launch"], count=12)
                if mutation == "partial":
                    overrides["campaign_status"] = "duration_cap_reached"
                if mutation == "bad_check":
                    overrides["correctness"] = None
                if mutation == "bad_sample":
                    overrides["raw_samples"] = []
                if mutation == "bad_cycles":
                    overrides["raw_samples"] = [{"event_ms": 1, "numerical_check": "passed",
                                                 "cycles_per_iteration": []}] * 2
                record = save_run(root, item, **overrides)
                if mutation == "hash":
                    record["attempts"][-1]["run_sha256"] = "wrong"
                if mutation == "missing_hash":
                    del record["attempts"][-1]["run_sha256"]
                write_reports(root, self.manifest([item]), {"jobs": {"test": record}})
                row = self.summary(root)["configurations"][0]
                self.assertEqual(row["status"], "invalid")
                self.assertNotIn("event_ms_median", row)

    def test_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.mkdir()
            (outside / "run.json").write_text('{}')
            (root / "escape").symlink_to(outside, target_is_directory=True)
            for path in ("../outside", str(outside), "escape"):
                with self.subTest(path=path):
                    record = {"status": "passed", "attempts": [{"status": "passed", "path": path,
                                                                "run_sha256": "ignored"}]}
                    write_reports(root, self.manifest([job("test")]), {"jobs": {"test": record}})
                    row = self.summary(root)["configurations"][0]
                    self.assertEqual(row["status"], "invalid")
                    self.assertEqual(row["error"], "Unsafe path or unreadable run.json")
            (root / "summary.json").unlink()
            sentinel = outside / "sentinel"
            sentinel.write_text("unchanged")
            (root / "summary.json").symlink_to(sentinel)
            with self.assertRaises(ValueError):
                write_reports(root, self.manifest([job("test")]), {"jobs": {}})
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_report_authenticates_full_evidence(self):
        for mutation in ("proof", "device", "snapshot", "output", "work", "rates", "normalized"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                item = job("test")
                record = save_run(root, item)
                attempt = record["attempts"][-1]
                directory = root / attempt["path"]
                path = directory / "run.json"
                run = json.loads(path.read_text())
                if mutation == "proof":
                    run["correctness"]["launch"]["repetitions"] = 3
                elif mutation == "device":
                    run["device"]["sm_count"] += 1
                elif mutation == "snapshot":
                    (directory / "host-source-snapshot/common/runner.py").write_text("changed")
                elif mutation == "output":
                    (directory / "first-timing-output.bin").write_bytes(b"changed")
                elif mutation == "normalized":
                    run["raw_samples"][0]["cycles_per_iteration"][0] += 1
                else:
                    run["raw_samples"][0][mutation] = {}
                path.write_text(json.dumps(run))
                attempt["run_sha256"] = sha256(path.read_bytes()).hexdigest()
                write_reports(root, self.manifest([item]), {"jobs": {"test": record}})
                self.assertEqual(self.summary(root)["configurations"][0]["status"], "invalid")

    def test_partial_nonfinite_json_is_not_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = job("failed")
            record = save_run(root, item, status="failed", raw_samples=[{"event_ms": float("nan")}])
            write_reports(root, self.manifest([item]), {"jobs": {"failed": record}})
            self.assertEqual(self.summary(root)["configurations"][0]["recorded_samples"], 0)
            with (root / "samples.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_throughput_units_and_equal_byte_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [job("bulk", kind="throughput", size=1024, blocks=2),
                     job("shape1", kind="throughput", size=16384, blocks=4,
                         case="tma_equal_bytes_shape_sweep", shape=[16, 16, 16]),
                     job("shape2", kind="throughput", size=16384, blocks=4,
                         case="tma_equal_bytes_shape_sweep", shape=[64, 64, 1])]
            write_reports(root, self.manifest(items), {"jobs": {item["id"]: save_run(root, item)
                                                              for item in items}})
            rows = {row["job_id"]: row for row in self.summary(root)["configurations"]}
            expected = ((2 * 8 * 1024 / 2 / 1e6) + (2 * 8 * 1024 / 3 / 1e6)) / 2
            self.assertAlmostEqual(rows["bulk"]["requested_gbps_median"], expected)
            svg = (root / "throughput.svg").read_text()
            self.assertIn("16x16x16", svg)
            self.assertIn("64x64x1", svg)
            self.assertIn("not distinct HBM bandwidth", svg)
            self.assertIn("2S", svg)
            self.assertEqual(svg.count("<circle"), 3)
            ET.fromstring(svg)

    def test_latest_attempt_and_missing_points_do_not_connect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [job("one", size=1024), job("gap", size=2048), job("three", size=4096)]
            records = {item["id"]: save_run(root, item) for item in items if item["id"] != "gap"}
            write_reports(root, self.manifest(items), {"jobs": records})
            svg = ET.parse(root / "latency.svg").getroot()
            lines = svg.findall("{http://www.w3.org/2000/svg}line")
            self.assertFalse(any(line.attrib.get("stroke-width") == "1.6" for line in lines))
            records["one"]["attempts"].append({"status": "running", "path": "runs/one/attempt-002"})
            records["one"]["status"] = "running"
            write_reports(root, self.manifest(items), {"jobs": records})
            rows = self.summary(root)["configurations"]
            self.assertEqual(rows[0]["status"], "running")
            self.assertNotIn("event_ms_median", rows[0])


if __name__ == "__main__":
    unittest.main()
