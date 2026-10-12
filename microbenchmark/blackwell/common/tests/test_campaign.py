"""Campaign lifecycle tests using synthetic workers, never GPU evidence."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from microbenchmark.blackwell import campaign
from microbenchmark.blackwell.common.metrics import rates
from microbenchmark.blackwell.common.types import Launch


DEVICE = {"name": "Synthetic B200", "ordinal": 0, "driver_version": 13000,
          "pci_bus_id": "0000:00:00.0", "compute_capability": [10, 0],
          "sm_count": 148, "l2_bytes": 126 * 1024**2,
          "gpu_uuid": "GPU-synthetic", "host": "synthetic-host"}
OPTIONS = {"samples": 2, "seed": 17, "global_bytes": 512 * 1024**2,
           "max_seconds": 60, "worker_timeout": 300}
SOURCE_BLOBS = {"common/runner.py": b"synthetic common source\n",
                "tma/run.py": b"synthetic family source\n"}
SOURCES = {name: campaign.sha256(blob) for name, blob in SOURCE_BLOBS.items()}


def small_manifest():
    """Keep lifecycle tests at two measurements, without input buffers."""
    artifacts = {f"{family}/{variant.id}": {"candidate_sha256": "c" * 64}
                 for family, (hooks, _) in campaign.families("tma").items()
                 for variant in hooks.variants()}
    with patch.object(campaign, "source_hashes", return_value=SOURCES):
        manifest = campaign.manifest_for("tma", deepcopy(DEVICE), deepcopy(OPTIONS), artifacts)
    manifest["jobs"] = manifest["jobs"][:2]
    manifest.pop("identity")
    manifest["identity"] = campaign.sha256(campaign.canonical(manifest))
    return manifest


def fake_report(job, directory, manifest):
    """Write a structurally valid report, explicitly with synthetic raw bytes."""
    hooks, _ = campaign.families(manifest["suite"])[job["family"]]
    variant = next(v for v in hooks.variants() if v.id == job["variant"]["id"])
    launch = Launch(**job["launch"])
    workload = hooks.work(variant, launch)
    output = bytes(job["memory"]["output_bytes"])
    sources = {name: digest for name, digest in manifest["sources"].items()
               if len(Path(name).parts) == 2 and name.endswith(".py")
               and Path(name).parts[0] in ("common", job["family"])}
    for name in sources:
        path = directory / "host-source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCE_BLOBS[name])
    (directory / "correctness-output.bin").write_bytes(output)
    (directory / "first-timing-output.bin").write_bytes(output)
    cycles = [1234] * launch.blocks
    sample = {"event_ms": 2.5, "numerical_check": "passed", "cycles": cycles,
              "cycles_per_iteration": [c / launch.repetitions for c in cycles],
              "work": workload, "rates": rates(workload, 2.5),
              "output_sha256": campaign.sha256(output)}
    report = {"variant": job["variant"], "launch": asdict(launch),
              "candidate_sha256": job["candidate_sha256"], "device": manifest["device"],
              "campaign_status": "requested_batches_completed",
              "host_source_hashes": sources,
              "correctness": {"passed": True, "launch": dict(job["launch"], repetitions=min(2, launch.repetitions)),
                              "output_sha256": campaign.sha256(output)},
              "correctness_output_sha256": campaign.sha256(output),
              "raw_samples": [deepcopy(sample) for _ in range(job["samples"])]}
    (directory / "run.json").write_text(json.dumps(report))
    return report


class CampaignPlanTests(unittest.TestCase):
    def test_all_default_configurations_without_duplicate_smoke_phase(self):
        jobs = campaign.plan_jobs("tma", DEVICE, OPTIONS)
        self.assertEqual(len(jobs), 216)
        self.assertTrue(all("phase" not in job for job in jobs))
        self.assertTrue(all(job["samples"] == OPTIONS["samples"] for job in jobs))
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        self.assertEqual(len({(j["family"], j["variant"]["id"]) for j in jobs}), 31)
        self.assertEqual(sum(j["family"] == "memory" for j in jobs), 21)
        self.assertEqual(sum(j["kind"] == "latency" for j in jobs), 42)
        throughput = [j for j in jobs if j["kind"] == "throughput"]
        self.assertEqual(len(throughput), 174)
        self.assertEqual({j["launch"]["blocks"] // DEVICE["sm_count"] for j in throughput}, set(range(1, 7)))
        for job in jobs:
            hooks, _ = campaign.families("tma")[job["family"]]
            variant = next(v for v in hooks.variants() if v.id == job["variant"]["id"])
            self.assertEqual(job["work"], hooks.work(variant, Launch(**job["launch"])))

    def test_offline_planning_does_not_touch_cuda_or_artifacts(self):
        with patch.object(campaign, "preflight", side_effect=AssertionError("GPU access")), \
             patch.object(campaign, "authenticate", side_effect=AssertionError("Artifact access")), \
             patch.object(campaign, "probe_device", side_effect=AssertionError("GPU access")), \
             patch("sys.argv", ["campaign", "--dry-run", "--sm-count", "148", "--l2-bytes", str(DEVICE["l2_bytes"])]), \
             redirect_stdout(io.StringIO()) as stream:
            campaign.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["hardware_execution"], "NOT_RUN")
        self.assertNotIn("smoke_configurations", result)
        self.assertEqual(result["measurement_configurations"], 216)

    def test_invalid_geometry_rejected(self):
        for change in ({"sm_count": 0}, {"sm_count": True}, {"l2_bytes": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                campaign.plan_jobs("tma", dict(DEVICE, **change), OPTIONS)
        with self.assertRaises(ValueError):
            campaign.plan_jobs("tma", DEVICE, dict(OPTIONS, global_bytes=DEVICE["l2_bytes"]))

    def test_identity_covers_source_device_options_and_artifacts(self):
        with patch.object(campaign, "source_hashes", return_value=SOURCES):
            baseline = campaign.manifest_for("tma", DEVICE, OPTIONS)
            altered = [campaign.manifest_for("tma", dict(DEVICE, gpu_uuid="GPU-other"), OPTIONS),
                       campaign.manifest_for("tma", DEVICE, dict(OPTIONS, samples=3)),
                       campaign.manifest_for("tma", DEVICE, dict(OPTIONS, seed=19))]
        with patch.object(campaign, "source_hashes", return_value={"common/runner.py": "changed"}):
            altered.append(campaign.manifest_for("tma", DEVICE, OPTIONS))
        self.assertTrue(all(m["identity"] != baseline["identity"] for m in altered))
        full = small_manifest()
        self.assertEqual(full["schema"], "blackwell-campaign-v2")
        changed = deepcopy(full)
        changed["artifacts"][next(iter(changed["artifacts"]))]["candidate_sha256"] = "d" * 64
        self.assertNotEqual(campaign.canonical(full), campaign.canonical(changed))


class CampaignLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.manifest = small_manifest()
        self.state = campaign.fresh_state(self.manifest)
        self.stack.enter_context(patch.object(campaign, "source_hashes", return_value=self.manifest["sources"]))
        self.stack.enter_context(patch.object(campaign, "authenticate", return_value=self.manifest["artifacts"]))
        self.reports = self.stack.enter_context(patch("microbenchmark.blackwell.common.reporting.write_reports"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.calls = []

    def worker(self, job, directory, *unused):
        self.calls.append(job["id"])
        on_disk = json.loads((self.root / "state.json").read_text())
        self.assertEqual(on_disk["jobs"][job["id"]]["status"], "running")
        fake_report(job, directory, self.manifest)

    def execute(self, **kwargs):
        campaign.execute_campaign(self.root, self.manifest, self.state, {}, run_job=kwargs.pop("run_job", self.worker), **kwargs)

    def test_completed_cases_skipped_and_checkpoints_valid(self):
        self.execute()
        self.assertEqual(self.reports.call_count, 1)
        self.assertEqual(self.calls, [j["id"] for j in self.manifest["jobs"]])
        self.assertEqual(self.state["status"], "completed")
        self.assertFalse((self.root / "state.json.tmp").exists())
        campaign.validate_resume(self.root, self.manifest, self.state)
        self.execute()
        self.assertEqual(self.reports.call_count, 2)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(len(e["attempts"]) == 1 for e in self.state["jobs"].values()))

    def test_failure_stops_campaign_and_retry_preserves_attempt(self):
        def fail(job, directory, *unused):
            self.calls.append(job["id"])
            (directory / "failure.txt").write_text("synthetic failure")
            raise campaign.WorkerFailure("failed", "synthetic failure")
        with self.assertRaises(campaign.WorkerFailure):
            self.execute(run_job=fail)
        self.assertEqual(len(self.calls), 1)
        first, second = (self.state["jobs"][j["id"]] for j in self.manifest["jobs"])
        self.assertEqual((first["status"], second["status"]), ("failed", "pending"))
        self.assertEqual(self.reports.call_count, 1)
        with self.assertRaises(ValueError):
            self.execute()
        failed_path = self.root / first["attempts"][0]["path"]
        self.execute(retry_failed=True)
        self.assertEqual(self.state["status"], "completed")
        self.assertEqual([a["status"] for a in first["attempts"]], ["failed", "passed"])
        self.assertEqual((failed_path / "failure.txt").read_text(), "synthetic failure")

    def test_strict_verification_requires_complete_raw_evidence(self):
        with self.assertRaises(ValueError):
            campaign.validate_completed(self.root, self.manifest, self.state)
        self.execute()
        self.assertEqual(campaign.validate_completed(self.root, self.manifest, self.state),
                         {"passed_jobs": 2, "passed_samples": 4})
        entry = next(iter(self.state["jobs"].values()))
        output = self.root / entry["attempts"][-1]["path"] / "first-timing-output.bin"
        output.unlink()
        with self.assertRaises(FileNotFoundError):
            campaign.validate_completed(self.root, self.manifest, self.state)

    def test_verify_cli_does_not_regenerate_reports(self):
        self.execute()
        campaign.atomic_json(self.root / "manifest.json", self.manifest)
        self.reports.reset_mock()
        with patch("sys.argv", ["campaign", "--verify-only", "--output", str(self.root)]):
            campaign.main()
        self.reports.assert_not_called()

    def test_timeout_and_interrupt_stop_without_launching_next(self):
        for exception, expected in ((campaign.WorkerFailure("timeout", "synthetic timeout"), "timeout"),
                                    (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = campaign.fresh_state(self.manifest)
                worker = MagicMock(side_effect=exception)
                with self.assertRaises(type(exception)):
                    campaign.execute_campaign(root, self.manifest, state, {}, run_job=worker)
                self.assertEqual(worker.call_count, 1)
                first = state["jobs"][self.manifest["jobs"][0]["id"]]
                self.assertEqual(first["status"], expected)
                self.assertEqual(json.loads((root / "state.json").read_text())["status"], "stopped")

    def test_running_state_requires_explicit_recovery(self):
        first = self.state["jobs"][self.manifest["jobs"][0]["id"]]
        first.update(status="running", attempts=[{"path": "runs/interrupted/attempt-001", "status": "running"}])
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.calls, [])
        self.execute(retry_failed=True)
        self.assertEqual([a["status"] for a in first["attempts"]], ["interrupted", "passed"])

    def test_attempt_reserved_before_mkdir_interruption(self):
        first_job = self.manifest["jobs"][0]
        relative = Path("runs") / first_job["id"] / "attempt-001"
        original_mkdir = Path.mkdir

        def interrupt_mkdir(path, *args, **kwargs):
            if path == self.root / relative:
                saved = json.loads((self.root / "state.json").read_text())
                entry = saved["jobs"][first_job["id"]]
                self.assertEqual(entry["status"], "running")
                self.assertEqual(entry["attempts"][-1]["path"], relative.as_posix())
                raise KeyboardInterrupt()
            return original_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", interrupt_mkdir), self.assertRaises(KeyboardInterrupt):
            self.execute()
        first = self.state["jobs"][first_job["id"]]
        self.assertEqual(self.calls, [])
        self.assertEqual(first["status"], "interrupted")
        self.assertFalse((self.root / relative).exists())
        self.execute(retry_failed=True)
        self.assertEqual([a["status"] for a in first["attempts"]], ["interrupted", "passed"])
        self.assertTrue(first["attempts"][-1]["path"].endswith("attempt-002"))

    def test_preexisting_orphan_preserved_and_retry_uses_next_attempt(self):
        first_job = self.manifest["jobs"][0]
        orphan = self.root / "runs" / first_job["id"] / "attempt-001"
        orphan.mkdir(parents=True)
        (orphan / "evidence.txt").write_text("unrecorded prior attempt")
        with self.assertRaises(FileExistsError):
            self.execute()
        first = self.state["jobs"][first_job["id"]]
        self.assertEqual(first["status"], "failed")
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            self.execute()
        self.execute(retry_failed=True)
        self.assertEqual([a["status"] for a in first["attempts"]], ["failed", "passed"])
        self.assertTrue(first["attempts"][-1]["path"].endswith("attempt-002"))
        self.assertEqual((orphan / "evidence.txt").read_text(), "unrecorded prior attempt")

    def test_tampered_completed_report_is_rejected_not_rerun(self):
        self.execute()
        entry = self.state["jobs"][self.manifest["jobs"][0]["id"]]
        path = self.root / entry["attempts"][-1]["path"] / "run.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.execute(retry_failed=True)
        self.assertEqual(len(self.calls), 2)

    def test_source_change_blocks_before_worker_launch(self):
        with patch.object(campaign, "source_hashes", return_value={"changed": "source"}), self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state["status"], "blocked")

    def test_artifact_change_blocks_before_worker_launch(self):
        with patch.object(campaign, "authenticate", return_value={"changed": "artifact"}), self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state["status"], "blocked")

    def test_manifest_content_tampering_rejected(self):
        self.manifest["options"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_manifest_identity_mismatch_rejected(self):
        self.state["identity"] = "wrong"
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_main_rejects_changed_resume_device_source_or_options(self):
        campaign.atomic_json(self.root / "manifest.json", self.manifest)
        campaign.atomic_json(self.root / "state.json", self.state)
        argv = ["campaign", "--execute", "--resume", "--output", str(self.root),
                "--accept-unvalidated-hardware-protocols", "--samples", "2"]
        for category in ("device", "sources", "options"):
            changed = deepcopy(self.manifest)
            changed[category][next(iter(changed[category]))] = "changed"
            with self.subTest(category=category), patch("sys.argv", argv), \
                 patch.object(campaign, "preflight", return_value={"device": DEVICE, "free_bytes": 8 * 1024**3}), \
                 patch.object(campaign, "manifest_for", return_value=changed), \
                 patch.object(campaign, "execute_campaign") as execute, \
                 self.assertRaisesRegex(ValueError, "Resume identity changed"):
                campaign.main()
            execute.assert_not_called()

    def test_unacknowledged_resume_does_not_probe_gpu(self):
        first = self.state["jobs"][self.manifest["jobs"][0]["id"]]
        first.update(status="timeout", attempts=[{"path": "runs/prior/attempt-001", "status": "timeout"}])
        campaign.atomic_json(self.root / "manifest.json", self.manifest)
        campaign.atomic_json(self.root / "state.json", self.state)
        with patch("sys.argv", ["campaign", "--execute", "--resume", "--output", str(self.root),
                                "--accept-unvalidated-hardware-protocols", "--samples", "2"]), \
             patch.object(campaign, "preflight") as probe, self.assertRaises(ValueError):
            campaign.main()
        probe.assert_not_called()


class CampaignResultTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = small_manifest()
        self.job = self.manifest["jobs"][-1]
        self.attempt = {"path": "runs/example/attempt-001"}
        self.output = self.root / self.attempt["path"]
        self.output.mkdir(parents=True)
        self.report = fake_report(self.job, self.output, self.manifest)

    def validate(self):
        return campaign.validate_result(self.root, self.manifest, self.job, self.attempt)

    def test_complete_result_and_invalid_samples(self):
        self.assertEqual(len(self.validate()), 64)
        changes = [lambda r: r["raw_samples"].pop(),
                   lambda r: r.update(campaign_status="duration_cap_reached"),
                   lambda r: r["raw_samples"][0].update(event_ms=0),
                   lambda r: r["raw_samples"][0].update(numerical_check="failed"),
                   lambda r: r["raw_samples"][0].update(cycles=[float("nan")]),
                   lambda r: r["raw_samples"][0].update(cycles_per_iteration=[1]),
                   lambda r: r["raw_samples"][0].update(rates={}),
                   lambda r: r.update(candidate_sha256="wrong"),
                   lambda r: r["correctness"].update(passed=False),
                   lambda r: r["correctness"]["launch"].update(repetitions=3),
                   lambda r: r["device"].update(ordinal=1)]
        for index, change in enumerate(changes):
            report = deepcopy(self.report)
            change(report)
            (self.output / "run.json").write_text(json.dumps(report))
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate()

    def test_snapshot_and_timing_output_tampering_rejected(self):
        for relative in ("host-source-snapshot/common/runner.py", "first-timing-output.bin", "correctness-output.bin"):
            path = self.output / relative
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=relative), self.assertRaises(ValueError):
                self.validate()
            path.write_bytes(original)

    def test_unsafe_attempt_and_symlink_rejected(self):
        for path in ("../escape", str(self.output.resolve())):
            with self.subTest(path=path), self.assertRaises(ValueError):
                campaign.validate_result(self.root, self.manifest, self.job, {"path": path})
        report = self.output / "run.json"
        report.rename(self.output / "real.json")
        report.symlink_to("real.json")
        with self.assertRaises(ValueError):
            self.validate()


class CampaignProcessTests(unittest.TestCase):
    def test_worker_owns_actual_gpu_child_not_nested_watchdog(self):
        manifest = small_manifest()
        job = manifest["jobs"][0]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(campaign, "bounded_process", return_value=0) as process:
            campaign.worker(job, Path(directory), {"tma": Path("synthetic-qualified")}, OPTIONS, 0)
        argv = process.call_args.args[0]
        self.assertIn("--worker", argv)
        self.assertIn("--execute", argv)
        self.assertEqual(process.call_args.args[1], OPTIONS["worker_timeout"])

    def test_timeout_kills_and_reaps_entire_worker_group(self):
        process = MagicMock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired(["synthetic"], 1), 0]
        with patch.object(campaign.subprocess, "Popen", return_value=process) as popen, \
             patch.object(campaign.os, "killpg") as killpg, \
             self.assertRaises(campaign.WorkerFailure) as raised:
            campaign.bounded_process(["synthetic"], 1, None, None)
        self.assertEqual(raised.exception.status, "timeout")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(12345, signal.SIGKILL)
        self.assertEqual(process.wait.call_count, 2)

    def test_interrupt_also_kills_group_and_preserves_interrupt(self):
        process = MagicMock(pid=12345)
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch.object(campaign.subprocess, "Popen", return_value=process), \
             patch.object(campaign.os, "killpg") as killpg, self.assertRaises(KeyboardInterrupt):
            campaign.bounded_process(["synthetic"], 1, None, None)
        killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_campaign_lock_excludes_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with campaign.campaign_lock(root):
                with self.assertRaises(RuntimeError):
                    with campaign.campaign_lock(root):
                        self.fail("Second lock owner entered")
            with campaign.campaign_lock(root):
                pass


class CampaignArtifactTests(unittest.TestCase):
    def test_missing_family_rejected_without_reading_artifacts(self):
        with self.assertRaises(ValueError):
            campaign.authenticate("tma", {})

    def test_authentication_pins_candidate_and_sass(self):
        from microbenchmark.blackwell.tma.tests.test_tma import fixture
        variant = campaign.families("tma")["tma"][0].variants()[0]
        source = b"synthetic SASS source\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualified = root / "qualified"
            qualified.mkdir()
            artifact = fixture(qualified, variant, source)
            family = root / "family"
            (family / "sass").mkdir(parents=True)
            (family / "sass" / (variant.id + ".sass")).write_bytes(source)
            pinned = {"variants": [{"id": variant.id, "candidate_sha256": campaign.sha256(b"synthetic-not-a-cubin")}]}
            (family / "manifest.json").write_text(json.dumps(pinned))
            registry = {"tma": (SimpleNamespace(directory=family, variants=lambda: [variant]), None)}
            with patch.object(campaign, "families", return_value=registry):
                hashes = campaign.authenticate("tma", {"tma": qualified})
                self.assertEqual(hashes["tma/" + variant.id]["source_sha256"], campaign.sha256(source))
                self.assertEqual(hashes["tma/" + variant.id]["qualification_sha256"],
                                 campaign.sha256((artifact / "qualification.json").read_bytes()))
                pinned["variants"][0]["candidate_sha256"] = "wrong"
                (family / "manifest.json").write_text(json.dumps(pinned))
                with self.assertRaisesRegex(ValueError, "pinned baseline"):
                    campaign.authenticate("tma", {"tma": qualified})


if __name__ == "__main__":
    unittest.main()
