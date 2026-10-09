"""CPU-only tests. Synthetic driver outputs below are never GPU evidence."""
from dataclasses import replace
import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microbenchmark.blackwell.common.artifacts import authenticated_candidate, checked_files, sha256
from microbenchmark.blackwell.common.cuda_driver import Driver
from microbenchmark.blackwell.common.metrics import rates, summarize
from microbenchmark.blackwell.common.provenance import inspect as inspect_import
from microbenchmark.blackwell.common.sass import cfg, parse_source
from microbenchmark.blackwell.tma.catalog import variants
from microbenchmark.blackwell.tma.inputs import Launch, check, cycle, prepare, tma_tile, validate_launch
from microbenchmark.blackwell.tma.run import execute, work
from microbenchmark.blackwell.tma.seeds import generate
from microbenchmark.blackwell.tma.sweeps import plan

TMA = Path(__file__).resolve().parents[1]
REPO = TMA.parents[2]


def fake_output(variant, launch, data):
    p = variant.parameters
    output = bytearray(validate_launch(variant, launch)["output_bytes"])
    for block in range(launch.blocks):
        index, checksum = 0, 0
        for iteration in range(launch.repetitions):
            tile = tma_tile(data, p, launch.count, index if p["fine"] else (iteration + block) % launch.count)
            word = int.from_bytes(tile[:4], "little")
            if p["fine"]:
                index, checksum = word, word
            else:
                checksum ^= word
        offset = block * (32 + p["transfer_bytes"])
        struct.pack_into("<QII", output, offset, 123, checksum, 0)
        output[offset + 32:offset + 32 + len(tile)] = tile
    return bytes(output)


def fixture(root, variant, source):
    directory = root / variant.id
    directory.mkdir()
    (root / "sass").mkdir(exist_ok=True)
    (root / "sass" / (variant.id + ".sass")).write_bytes(source)
    (directory / "candidate.cubin").write_bytes(b"synthetic-not-a-cubin")
    (directory / "candidate.nvdisasm.json").write_bytes(b"[]")
    report = {"id": variant.id, "parameters": variant.parameters, "status": "sass_encoded",
              "candidate_sha256": sha256(b"synthetic-not-a-cubin"), "source_sha256": sha256(source),
              "disassembly_sha256": sha256(b"[]"), "unknowns": [], "static_errors": []}
    (directory / "qualification.json").write_text(json.dumps(report))
    return directory


class BaselineTests(unittest.TestCase):
    def test_exact_tma_catalog(self):
        rows = json.loads((TMA / "manifest.json").read_text())["variants"]
        actual = variants()
        self.assertEqual(len(actual), 30)
        self.assertEqual(len({v.id for v in actual}), 30)
        self.assertTrue(all(v.family == "tma" for v in actual))
        self.assertEqual([v.to_dict() for v in actual],
                         [{k: row[k] for k in ("id", "case", "family", "parameters")} for row in rows])

    def test_all_ptx_and_sass_hashes_unchanged(self):
        pinned = {r["id"]: r for r in json.loads((TMA / "manifest.json").read_text())["variants"]}
        self.assertEqual(len(list((TMA / "sass").glob("*.sass"))), 30)
        for variant in variants():
            with self.subTest(variant=variant.id):
                row = pinned[variant.id]
                self.assertEqual(sha256(generate(variant).encode()), row["ptx_sha256"])
                source = (TMA / "sass" / (variant.id + ".sass")).read_bytes()
                self.assertEqual(sha256(source), row["source_sha256"])
                self.assertEqual(len(parse_source(source.decode())), row["instruction_words"])

    def test_retained_tma_protocol(self):
        from microbenchmark.blackwell.tma.protocols import inspect
        for variant in variants():
            rows = parse_source((TMA / "sass" / (variant.id + ".sass")).read_text())
            self.assertFalse(inspect(variant, generate(variant), rows)["errors"])
            self.assertFalse(cfg(rows)["errors"])
            stripped = [r for r in rows if not r["sass"].lstrip().startswith(("UTMALDG", "UBLKCP"))]
            self.assertIn("TMA issue retained", inspect(variant, generate(variant), stripped)["errors"])

    def test_dry_run_needs_no_f2asm_or_cuda(self):
        script = """
import ctypes, runpy, sys
def forbidden(*args, **kwargs): raise AssertionError('CUDA library opened')
ctypes.CDLL = forbidden
sys.argv = ['run', '--variant', 'tma_tensor_3d_throughput--8x8x8', '--dry-run']
runpy.run_module('microbenchmark.blackwell.tma.run', run_name='__main__')
assert not any(n == 'f2asm' or n.startswith('f2asm.') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=REPO,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["hardware_execution"], "NOT_RUN")

    def test_run_requires_explicit_acknowledgement(self):
        result = subprocess.run([sys.executable, "-B", "-m", "microbenchmark.blackwell.tma.run",
                                 "--variant", "tma_completion_latency", "--execute"],
                                cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--accept-unvalidated-hardware-protocols", result.stderr)


class InputTests(unittest.TestCase):
    def test_randomized_closed_chain(self):
        for count in (1, 2, 7, 31):
            nxt = cycle(count, 17)
            seen, node = set(), 0
            for _ in range(count):
                self.assertNotIn(node, seen)
                seen.add(node)
                node = nxt[node]
            self.assertEqual(node, 0)
            self.assertEqual(len(seen), count)

    def test_all_tiles_and_corruption(self):
        for variant in variants():
            with self.subTest(variant=variant.id):
                launch = Launch(count=3, repetitions=5, blocks=1 if variant.parameters["fine"] else 2)
                data, sizes = prepare(variant, launch)
                output = fake_output(variant, launch, data)
                self.assertEqual(len(data), sizes["input_bytes"])
                self.assertTrue(check(variant, launch, output, data)["passed"])
                for offset in (0, 8, 12, len(output) - 1):
                    bad = bytearray(output)
                    if offset == 0:
                        bad[:8] = bytes(8)
                    else:
                        bad[offset] ^= 1
                    with self.assertRaises(ValueError):
                        check(variant, launch, bad, data)

    def test_tensor_coordinate_layout(self):
        p = {"shape": [2, 2, 2], "element_bytes": 4, "transfer_bytes": 32}
        data = struct.pack("<24I", *range(24))
        expected = struct.pack("<8I", 2, 3, 8, 9, 14, 15, 20, 21)
        self.assertEqual(tma_tile(data, p, 3, 1), expected)

    def test_launch_bounds(self):
        variant = variants()[0]
        for launch in (Launch(blocks=2), Launch(count=0), Launch(repetitions=0),
                       Launch(count=1 << 25), Launch(memory_limit=1), Launch(blocks=True)):
            with self.assertRaises(ValueError):
                validate_launch(variant, launch)

    def test_sweep_and_requested_traffic(self):
        # Synthetic geometry is test input, never device evidence.
        for variant in variants():
            jobs = plan(variant, 8, l2_bytes=1 << 20, global_bytes=1 << 24)
            self.assertTrue(jobs)
            self.assertTrue(all(job["status"] == "not_run" for job in jobs))
            if not variant.parameters["fine"]:
                self.assertEqual([j["launch"]["blocks"] for j in jobs], [8, 16, 24, 32])
                for job in jobs:
                    launch = Launch(**job["launch"])
                    self.assertGreaterEqual(work(variant, launch)["transferred_bytes"], 4_000_000_000)


class ArtifactTests(unittest.TestCase):
    def test_candidate_tamper_rejected(self):
        variant = variants()[0]
        source = (TMA / "sass" / (variant.id + ".sass")).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = fixture(root, variant, source)
            source_path = root / "sass" / (variant.id + ".sass")
            self.assertEqual(authenticated_candidate(directory, variant, source_path)[0], b"synthetic-not-a-cubin")
            (directory / "candidate.cubin").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                authenticated_candidate(directory, variant, source_path)

    def test_source_and_disassembly_tamper_rejected(self):
        variant = variants()[0]
        source = (TMA / "sass" / (variant.id + ".sass")).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = fixture(root, variant, source)
            chosen = root / "chosen.sass"
            chosen.write_bytes(source + b"# changed\n")
            with self.assertRaises(ValueError):
                authenticated_candidate(directory, variant, chosen)
            chosen.write_bytes(source)
            (directory / "candidate.nvdisasm.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                authenticated_candidate(directory, variant, chosen)

    def test_provenance_paths_and_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_bytes(b"original")
            record = {"source_commit": "test", "source_files": {},
                      "files": {"a": {"sha256": sha256(b"original")}}}
            self.assertTrue(inspect_import(record, root)["clean"])
            (root / "a").write_bytes(b"local fix")
            self.assertEqual(inspect_import(record, root)["destination_changes"], ["a"])
            with self.assertRaises(ValueError):
                checked_files(root, {"../outside": "irrelevant"})
            with self.assertRaises(ValueError):
                inspect_import({**record, "files": {"../outside": {"sha256": "x"}}}, root)


class HostTests(unittest.TestCase):
    def test_argument_abi(self):
        driver = Driver.__new__(Driver)
        captured = []
        def call(name, *args):
            values = [C.cast(args[9][i], C.POINTER(kind)).contents.value
                      for i, kind in enumerate((C.c_uint64, C.c_uint64, C.c_uint, C.c_uint, C.c_uint64))]
            captured.append((name, args[1:8], values))
        driver.call = call
        driver.launch(None, 2, 32, 0x100000000, 0x200000000, 7, 3, 0x300000000)
        self.assertEqual(captured, [("cuLaunchKernel", (2, 1, 1, 32, 1, 1, 0),
                                     [0x100000000, 0x200000000, 7, 3, 0x300000000])])

    def test_tensor_descriptor_layout(self):
        driver = Driver.__new__(Driver)
        captured = []
        def call(name, *args):
            captured.append((name, args[0] % 64, list(args[4]), list(args[5]), list(args[6])))
        driver.call, driver.upload = call, lambda data: len(data)
        self.assertEqual(driver.tensor_map(0x1000, [8, 8, 8], 3, 4), 128)
        self.assertEqual(captured, [("cuTensorMapEncodeTiled", 0, [24, 8, 8], [96, 768], [8, 8, 8])])

    def test_synthetic_host_workflow(self):
        variant = next(v for v in variants() if v.id == "tma_tensor_3d_throughput--8x8x8")
        launch = Launch(count=3, repetitions=3)
        class FakeDriver:
            def __init__(self, device):
                self.info = {"pci_bus_id": "SYNTHETIC", "name": "FAKE CPU TEST"}
                self.buffers, self.calls, self.closed = {}, [], False
            def upload(self, data):
                pointer = len(self.buffers) + 1
                self.buffers[pointer] = data
                return pointer
            def tensor_map(self, *args): return 3
            def load(self, binary): return "FAKE_FUNCTION"
            def launch(self, function, blocks, threads, inp, out, repetitions, count, tensor):
                self.calls.append(repetitions)
                current = replace(launch, blocks=blocks, repetitions=repetitions, count=count)
                self.buffers[out] = fake_output(variant, current, self.buffers[inp])
            def call(self, *args): pass
            def download(self, pointer, size): return self.buffers[pointer]
            def timed(self, callback): callback(); return 1.0
            def close(self): self.closed = True
        driver = FakeDriver(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qualified, output = root / "qualified", root / "output"
            qualified.mkdir(); output.mkdir()
            fixture(qualified, variant, (TMA / "sass" / (variant.id + ".sass")).read_bytes())
            args = SimpleNamespace(qualified=qualified, device=0, samples=2, max_seconds=10)
            with patch("microbenchmark.blackwell.tma.run.telemetry", return_value={"synthetic": True}):
                report = execute(args, variant, launch, output, lambda _: driver)
            self.assertEqual(driver.calls, [2, 3, 3])
            self.assertTrue(driver.closed)
            self.assertEqual(len(report["raw_samples"]), 2)
            self.assertFalse(report["performance_claim_eligible"])
            self.assertEqual(report["campaign_status"], "requested_batches_completed")
            self.assertTrue((output / "run.json").is_file())

    def test_unit_explicit_statistics(self):
        self.assertEqual(rates({"transferred_bytes": 1024}, 1)["transferred_bytes_per_second"], 1024000)
        self.assertEqual(summarize([1, 2, 3])["median"], 2)
        with self.assertRaises(ValueError):
            rates({"transferred_bytes": 1}, 0)


if __name__ == "__main__":
    unittest.main()
