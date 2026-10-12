import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..publication import export, load
from ...campaign import fresh_state, sha256
from ...common.metrics import summarize
from ...common.tests.test_campaign import small_manifest, fake_report


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.public = self.root / "raw", self.root / "public"
        self.raw.mkdir()
        manifest = small_manifest()
        state = fresh_state(manifest)
        state["status"] = "completed"
        for job in manifest["jobs"]:
            path = Path("runs") / job["id"] / "attempt-001"
            directory = self.raw / path
            directory.mkdir(parents=True)
            fake_report(job, directory, manifest)
            state["jobs"][job["id"]] = {"status": "passed", "attempts": [
                {"status": "passed", "path": str(path), "run_sha256": sha256((directory / "run.json").read_bytes())}]}
        (self.raw / "manifest.json").write_text(json.dumps(manifest))
        (self.raw / "state.json").write_text(json.dumps(state))

    def test_export_verifies_samples_and_removes_identity(self):
        self.assertEqual(export(self.raw, self.public), {"passed_jobs": 2, "passed_samples": 4})
        metadata, rows, samples = load(self.public)
        self.assertEqual((len(rows), len(samples)), (2, 4))
        for name in ("provenance.json", "summary.json"):
            text = (self.public / name).read_text()
            self.assertNotIn("GPU-synthetic", text)
            self.assertNotIn("synthetic-host", text)
            self.assertNotIn("pci_bus_id", text)
            self.assertNotIn(str(self.root), text)
        self.assertFalse(metadata["performance_claim_eligible"])

    def test_missing_raw_evidence_prevents_export(self):
        next(self.raw.glob("runs/*/*/first-timing-output.bin")).unlink()
        with self.assertRaises(FileNotFoundError):
            export(self.raw, self.public)
        self.assertFalse(self.public.exists())

    def test_tampering_is_rejected_even_with_updated_sample_hash(self):
        export(self.raw, self.public)
        path = self.public / "samples.csv.gz"
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load(self.public)
        text = gzip.decompress(original).decode().replace(",2.5,", ",3.5,", 1)
        self.assertNotEqual(text, gzip.decompress(original).decode())
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
        metadata_path = self.public / "provenance.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["file_hashes"][path.name] = sha256(path.read_bytes())
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "differs from individual samples"):
            load(self.public)

    def test_export_does_not_overwrite_existing_directory(self):
        export(self.raw, self.public)
        with self.assertRaises(FileExistsError):
            export(self.raw, self.public)

    def test_cross_python_statistical_roundoff_is_tolerated(self):
        export(self.raw, self.public)
        def rounded(values):
            result = summarize(values)
            result["mean"] = math.nextafter(result["mean"], math.inf)
            return result
        with patch("microbenchmark.blackwell.plots.publication.summarize", side_effect=rounded):
            load(self.public)
