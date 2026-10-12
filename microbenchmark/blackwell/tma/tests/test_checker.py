"""Exact lightweight CPU oracles, compared with independent transfer replay."""
from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

from .. import inputs
from ..catalog import variants
from .test_tma import fake_output


class CheckerTests(unittest.TestCase):
    def test_chunked_input_preserves_seeded_bytes(self):
        variant = next(v for v in variants() if v.id == "tma_bulk_throughput--1024")
        for count in (1, 1024, 1025, 2049):
            launch = inputs.Launch(count=count)
            data, sizes = inputs.prepare(variant, launch)
            self.assertEqual(data, inputs.random.Random(launch.seed).randbytes(sizes["input_bytes"]))

    def test_full_buffer_uses_bounded_random_requests(self):
        variant = next(v for v in variants() if v.id == "tma_bulk_throughput--1024")
        launch = inputs.Launch(count=1 << 19, memory_limit=1 << 30)
        # Exercise the actual 512-MiB plan without allocating it in unit tests.
        with patch.object(inputs.random.Random, "randbytes", return_value=b"") as fill:
            _, sizes = inputs.prepare(variant, launch)
        requests = [call.args[0] for call in fill.call_args_list]
        self.assertEqual(sum(requests), sizes["input_bytes"])
        self.assertEqual(sizes["input_bytes"], 512 * 1024**2)
        self.assertLessEqual(max(requests), 1 << 20)

    def test_every_variant_matches_transfer_replay(self):
        for variant in variants():
            for count in (1, 3, 7):
                for repetitions in sorted({1, max(1, count - 1), count,
                                           count + 1, 2 * count, 3 * count - 1}):
                    with self.subTest(variant=variant.id, count=count, repetitions=repetitions):
                        launch = inputs.Launch(count=count, repetitions=repetitions,
                                               blocks=1 if variant.parameters["fine"] else count + 2)
                        data, _ = inputs.prepare(variant, launch)
                        expected = fake_output(variant, launch, data)
                        checker = inputs.make_checker(variant, launch, data)
                        self.assertTrue(checker(expected)["passed"])
                        self.assertTrue(checker(expected)["passed"])

    def test_maximum_repetitions_do_not_replay_each_transfer(self):
        variant = next(v for v in variants() if v.id == "tma_tensor_3d_throughput--8x8x8")
        launch = inputs.Launch(count=5, repetitions=1 << 24, blocks=128)
        data, sizes = inputs.prepare(variant, launch)
        # A whole even number of cycles cancels. Brute-force the equivalent
        # short sequence for each CTA rather than use the rolling-window code.
        equivalent = replace(launch, repetitions=launch.repetitions % (2 * launch.count))
        expected = fake_output(variant, equivalent, data)
        self.assertEqual(len(expected), sizes["output_bytes"])
        with patch.object(inputs.struct, "unpack_from", wraps=struct.unpack_from) as reads, \
                patch.object(inputs, "tma_tile", wraps=inputs.tma_tile) as tiles:
            checker = inputs.make_checker(variant, launch, data)
            self.assertLessEqual(reads.call_count, launch.count + 2 * (launch.blocks - 1))
            self.assertEqual(tiles.call_count, min(launch.count, launch.blocks))
        # Once prepared, samples need neither input simulation nor tile gathers.
        with patch.object(inputs, "tma_tile", side_effect=AssertionError("Oracle recomputed")):
            for _ in range(5):
                self.assertTrue(checker(expected)["passed"])

    def test_cached_checker_still_rejects_corruption(self):
        for variant in variants():
            launch = inputs.Launch(count=7, repetitions=11,
                                   blocks=1 if variant.parameters["fine"] else 3)
            data, _ = inputs.prepare(variant, launch)
            output = fake_output(variant, launch, data)
            checker = inputs.make_checker(variant, launch, data)
            record = 32 + variant.parameters["transfer_bytes"]
            for block in range(launch.blocks):
                for offset in (0, 8, 12, 32, record - 1):
                    with self.subTest(variant=variant.id, block=block, offset=offset):
                        corrupted = bytearray(output)
                        start = block * record + offset
                        if offset == 0:
                            corrupted[start:start + 8] = bytes(8)
                        else:
                            corrupted[start] ^= 1
                        with self.assertRaises(ValueError):
                            checker(corrupted)
            with self.assertRaises(ValueError):
                checker(output[:-1])
            self.assertTrue(checker(output)["passed"])

    def test_oracles_are_scoped_to_input_and_launch(self):
        variant = next(v for v in variants() if v.id == "tma_tensor_3d_throughput--8x8x8")
        launch = inputs.Launch(count=7, repetitions=3, blocks=2)
        data, _ = inputs.prepare(variant, launch)
        different_launch = replace(launch, repetitions=4)
        different_input, _ = inputs.prepare(variant, replace(launch, seed=91))
        checker = inputs.make_checker(variant, launch, data)
        for other_launch, other_data in ((different_launch, data), (launch, different_input)):
            raw = fake_output(variant, other_launch, other_data)
            with self.assertRaises(ValueError):
                checker(raw)
            self.assertTrue(inputs.make_checker(variant, other_launch, other_data)(raw)["passed"])

    def test_bad_input_is_rejected_before_sampling(self):
        variant = next(v for v in variants() if v.parameters["fine"])
        launch = inputs.Launch(count=3, repetitions=2)
        data, _ = inputs.prepare(variant, launch)
        with self.assertRaisesRegex(ValueError, "Input length"):
            inputs.make_checker(variant, launch, data[:-1])
        corrupted = bytearray(data)
        struct.pack_into("<I", corrupted, 0, launch.count)
        with self.assertRaisesRegex(ValueError, "pointer escaped"):
            inputs.make_checker(variant, launch, corrupted)


if __name__ == "__main__":
    unittest.main()
