"""CPU-only qualification: no synthetic output below is hardware evidence."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microbenchmark.blackwell.common.artifacts import authenticated_candidate, sha256
from microbenchmark.blackwell.common.build import snapshot_sources
from microbenchmark.blackwell.common.chains import cycle
from microbenchmark.blackwell.common.sass import parse_source
from microbenchmark.blackwell.memory.build import suite
from microbenchmark.blackwell.memory.catalog import variants
from microbenchmark.blackwell.memory.inputs import Launch, check, final_index, make_checker, prepare, validate_launch
from microbenchmark.blackwell.memory.protocols import inspect
from microbenchmark.blackwell.memory.run import execute
from microbenchmark.blackwell.memory.seeds import generate
from microbenchmark.blackwell.memory.sweeps import plan
from microbenchmark.blackwell.memory.tests.sass_model import interpret

MEMORY = Path(__file__).resolve().parents[1]
REPO = MEMORY.parents[2]
VARIANT = variants()[0]
SOURCE = MEMORY / "sass" / (VARIANT.id + ".sass")


def output(launch, data):
    result = bytearray(32)
    struct.pack_into("<QIII", result, 0, 1230, final_index(VARIANT, launch, data), 0, 0)
    return bytes(result)


def fixture(root):
    directory = root / VARIANT.id
    directory.mkdir(parents=True)
    (root / "sass").mkdir()
    (root / "sass" / SOURCE.name).write_bytes(SOURCE.read_bytes())
    binary, disassembly = b"synthetic-not-a-cubin", b"[]"
    (directory / "candidate.cubin").write_bytes(binary)
    (directory / "candidate.nvdisasm.json").write_bytes(disassembly)
    row = {**VARIANT.to_dict(), "status": "sass_encoded", "static_errors": [], "unknowns": [],
           "candidate_sha256": sha256(binary), "source_sha256": sha256(SOURCE.read_bytes()),
           "disassembly_sha256": sha256(disassembly)}
    (directory / "qualification.json").write_text(json.dumps(row))
    return directory


class SourceTests(unittest.TestCase):
    def test_pinned_sources(self):
        row = json.loads((MEMORY / "manifest.json").read_text())["variants"][0]
        self.assertEqual({k: row[k] for k in ("id", "case", "family", "parameters")}, VARIANT.to_dict())
        self.assertEqual(sha256(SOURCE.read_bytes()), row["source_sha256"])
        self.assertEqual(sha256(generate(VARIANT).encode()), row["ptx_sha256"])
        self.assertEqual(len(parse_source(SOURCE.read_text())), row["instruction_words"])

    def test_retained_protocol_and_removed_work(self):
        rows = parse_source(SOURCE.read_text())
        self.assertEqual(inspect(VARIANT, generate(VARIANT), rows)["errors"], [])
        changed = [dict(row) for row in rows]
        changed[0x3e0 // 16]["sass"] = "NOP;"
        self.assertIn("retained dependent measured loads", inspect(VARIANT, generate(VARIANT), changed)["errors"])

    def test_sass_data_flow_and_exact_dynamic_load_counts(self):
        rows = parse_source(SOURCE.read_text())
        # Include all unroll/tail boundaries, full cycles, and non-multiples.
        for count in (1, 2, 3, 4, 5, 7, 11, 31):
            for repetitions in (1, 2, 3, 4, 5, 7, 8, 9, 16, 33):
                for seed in (0, 17):
                    with self.subTest(count=count, repetitions=repetitions, seed=seed):
                        launch = Launch(count=count, repetitions=repetitions, seed=seed)
                        data, _ = prepare(VARIANT, launch)
                        model = interpret(rows, data, count, repetitions)
                        self.assertEqual(model["clock_load_counts"], [count, count + repetitions])
                        self.assertEqual(model["final_index"], final_index(VARIANT, launch, data))
                        node, expected = 0, []
                        chain = cycle(count, seed)
                        for _ in range(count + repetitions):
                            expected.append(node)
                            node = chain[node]
                        self.assertEqual(model["indices"], expected)

    def test_changed_dependency_rejected_by_data_flow(self):
        rows = parse_source(SOURCE.read_text())
        rows[0x3f0 // 16]["sass"] = "IMAD.WIDE.U32 R6, RZ, 0x20, R10;"
        launch = Launch(count=11, repetitions=23, seed=17)
        data, _ = prepare(VARIANT, launch)
        model = interpret(rows, data, launch.count, launch.repetitions)
        self.assertNotEqual(model["final_index"], final_index(VARIANT, launch, data))

    def test_build_snapshot_has_no_tma_dependency(self):
        sources = snapshot_sources(suite())
        self.assertIn("memory/seeds/" + VARIANT.id + ".ptx", sources)
        self.assertIn("common/build.py", sources)
        self.assertFalse(any(name.startswith("tma/") for name in sources))


class InputTests(unittest.TestCase):
    def test_cycle_and_oracle(self):
        for count in (1, 3, 17):
            launch = Launch(count=count, repetitions=23, seed=17)
            data, sizes = prepare(VARIANT, launch)
            self.assertEqual(data, prepare(VARIANT, launch)[0])
            node, seen = 0, set()
            for _ in range(count):
                self.assertNotIn(node, seen)
                seen.add(node)
                node = struct.unpack_from("<I", data, node * 32)[0]
            self.assertEqual(node, 0)
            self.assertEqual(len(seen), count)
            self.assertEqual(sizes["threads"], 1)
            self.assertTrue(check(VARIANT, launch, output(launch, data), data)["passed"])

    def test_corrupt_results_rejected(self):
        launch = Launch(count=7, repetitions=9)
        data, _ = prepare(VARIANT, launch)
        raw = output(launch, data)
        for offset in (0, 8, 12):
            corrupted = bytearray(raw)
            if offset == 0:
                corrupted[:8] = bytes(8)
            else:
                corrupted[offset] ^= 1
            with self.assertRaises(ValueError):
                check(VARIANT, launch, corrupted, data)
        with self.assertRaises(ValueError):
            check(VARIANT, launch, raw[:-1], data)

    def test_reused_checker_computes_chain_once_and_rejects_corruption(self):
        launch = Launch(count=7, repetitions=9)
        data, _ = prepare(VARIANT, launch)
        raw = output(launch, data)
        with patch("microbenchmark.blackwell.memory.inputs.final_index", wraps=final_index) as oracle:
            checker = make_checker(VARIANT, launch, data)
            oracle.assert_called_once_with(VARIANT, launch, data)
            self.assertTrue(checker(raw)["passed"])
            for offset in (0, 8, 12):
                corrupted = bytearray(raw)
                if offset == 0:
                    corrupted[:8] = bytes(8)
                else:
                    corrupted[offset] ^= 1
                with self.assertRaises(ValueError):
                    checker(corrupted)
            with self.assertRaises(ValueError):
                checker(raw[:-1])
            self.assertTrue(checker(raw)["passed"])
            oracle.assert_called_once_with(VARIANT, launch, data)

    def test_invalid_launches(self):
        for launch in (Launch(blocks=2), Launch(count=0), Launch(repetitions=0),
                       Launch(count=1 << 25), Launch(repetitions=1 << 25),
                       Launch(memory_limit=16), Launch(blocks=True)):
            with self.assertRaises(ValueError):
                validate_launch(VARIANT, launch)

    def test_out_of_bounds_chain_rejected(self):
        launch = Launch(count=3, repetitions=1)
        data, _ = prepare(VARIANT, launch)
        corrupted = bytearray(data)
        struct.pack_into("<I", corrupted, 0, 3)
        with self.assertRaises(ValueError):
            final_index(VARIANT, launch, corrupted)

    def test_sweep_matches_tma_working_sets(self):
        from microbenchmark.blackwell.tma.catalog import variants as tma_variants
        from microbenchmark.blackwell.tma.sweeps import plan as tma_plan
        geometry = dict(sm_count=128, l2_bytes=64 * 1024 * 1024)
        ordinary = plan(VARIANT, **geometry)
        tma = tma_plan(tma_variants()[0], **geometry)
        self.assertEqual(len(ordinary), 21)
        self.assertEqual([j["memory"]["input_bytes"] for j in ordinary],
                         [j["memory"]["input_bytes"] for j in tma])
        self.assertTrue(all(j["launch"]["blocks"] == 1 and j["status"] == "not_run" for j in ordinary))
        self.assertTrue(all(j["launch"]["repetitions"] == 1048576 for j in ordinary))
        self.assertEqual(sum(len(tma_plan(v, **geometry)) for v in tma_variants()) + len(ordinary), 216)
        for arguments in (dict(sm_count=0, l2_bytes=1024), dict(sm_count=1, l2_bytes=1 << 30)):
            with self.assertRaises(ValueError):
                plan(VARIANT, **arguments)


class HostTests(unittest.TestCase):
    def test_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = fixture(Path(temp))
            authenticated_candidate(directory, VARIANT, SOURCE)
            (directory / "candidate.cubin").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                authenticated_candidate(directory, VARIANT, SOURCE)

    def test_synthetic_host_execution(self):
        launch = Launch(count=7, repetitions=9, seed=17)
        class FakeDriver:
            def __init__(self):
                self.info = {"pci_bus_id": "SYNTHETIC", "name": "CPU-ONLY FAKE"}
                self.buffers, self.calls, self.closed = {}, [], False
            def upload(self, data):
                pointer = len(self.buffers) + 1
                self.buffers[pointer] = data
                return pointer
            def load(self, binary): return "FAKE_FUNCTION"
            def tensor_map(self, *args): raise AssertionError("Ordinary LDG needs no tensor descriptor")
            def launch(self, function, blocks, threads, inp, out, repetitions, count, tensor):
                if (blocks, threads, tensor) != (1, 1, 0):
                    raise AssertionError("Wrong ordinary-load launch")
                self.calls.append(repetitions)
                self.buffers[out] = output(replace(launch, repetitions=repetitions), self.buffers[inp])
            def call(self, *args): pass
            def download(self, pointer, size): return self.buffers[pointer]
            def timed(self, callback): callback(); return 1.0
            def close(self): self.closed = True
        driver = FakeDriver()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "qualified")
            directory = root / "run"
            directory.mkdir()
            args = SimpleNamespace(qualified=root / "qualified", device=0, samples=2, max_seconds=10)
            with patch("microbenchmark.blackwell.memory.run.telemetry", return_value={"synthetic": True}), \
                    patch("microbenchmark.blackwell.memory.run.make_checker", wraps=make_checker) as factory:
                report = execute(args, VARIANT, launch, directory, lambda _: driver)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual([call.args[1].repetitions for call in factory.call_args_list], [2, 9])
            self.assertEqual(driver.calls, [2, 9, 9])
            self.assertTrue(driver.closed)
            self.assertEqual(report["campaign_status"], "requested_batches_completed")
            self.assertFalse(report["performance_claim_eligible"])
            self.assertEqual(report["raw_samples"][0]["cycles_per_iteration"], [1230 / 9])
            self.assertEqual(report["raw_samples"][0]["work"], {"dependent_loads": 9})
            self.assertIn("full_chain_warmup", report["event_metric"])
            self.assertIsInstance(report["cpu_oracle_setup_seconds"], float)
            self.assertGreaterEqual(report["cpu_oracle_setup_seconds"], 0)
            for sample in report["raw_samples"]:
                self.assertIsInstance(sample["cpu_check_seconds"], float)
                self.assertGreaterEqual(sample["cpu_check_seconds"], 0)
                self.assertEqual(sample["event_ms"], 1.0)
                self.assertEqual(sample["numerical_check"], "passed")

    def test_dry_run_no_cuda_or_assembler(self):
        script = """
import ctypes, runpy, sys
def forbidden(*a, **k): raise AssertionError('CUDA loaded')
ctypes.CDLL = forbidden
sys.argv = ['run', '--variant', 'memory_l1_pointer_chase', '--dry-run']
runpy.run_module('microbenchmark.blackwell.memory.run', run_name='__main__')
assert not any(n == 'f2asm' or n.startswith('f2asm.') for n in sys.modules)
assert not any(n.startswith('microbenchmark.blackwell.tma') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=REPO,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["hardware_execution"], "NOT_RUN")

    def test_explicit_hardware_acknowledgement_required(self):
        result = subprocess.run([sys.executable, "-B", "-m", "microbenchmark.blackwell.memory.run",
                                 "--variant", VARIANT.id, "--execute"],
                                cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--accept-unvalidated-hardware-protocols", result.stderr)


if __name__ == "__main__":
    unittest.main()
