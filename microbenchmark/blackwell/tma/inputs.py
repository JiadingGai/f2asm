"""TMA-only inputs, launch bounds and final-tile/checksum oracles; no CUDA dependency."""
from ..common.types import Launch
from ..common.chains import cycle
import random
import struct


def validate_launch(variant, launch):
    if variant.family != "tma":
        raise ValueError("TMA extraction only")
    for name in ("blocks", "repetitions", "count", "memory_limit"):
        value = getattr(launch, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if launch.blocks > 65535 or launch.repetitions > 1 << 24 or launch.count > 1 << 24:
        raise ValueError("Bounded-batch limit exceeded")
    p = variant.parameters
    if p["fine"] and launch.blocks != 1:
        raise ValueError("TMA dependent latency requires one CTA")
    if p["shape"] and launch.count * p["shape"][0] > 0x7fffffff:
        raise ValueError("Signed tensor-map coordinate would overflow")
    inp = launch.count * p["transfer_bytes"]
    out = launch.blocks * (32 + p["transfer_bytes"])
    if inp % 16:
        raise ValueError("TMA alignment")
    if inp + out > launch.memory_limit:
        raise ValueError("Memory budget exceeded before allocation")
    return {"input_bytes": inp, "output_bytes": out, "threads": 32,
            "hardware_correctness": "NOT_RUN"}


def tma_tile(data, parameters, count, index):
    shape = parameters["shape"]
    size = parameters["transfer_bytes"]
    if not shape:
        return data[index * size:(index + 1) * size]
    width = shape[0] * parameters["element_bytes"]
    rows, pitch = size // width, count * width
    return b"".join(data[r * pitch + index * width:r * pitch + (index + 1) * width]
                    for r in range(rows))


def prepare(variant, launch):
    sizes = validate_launch(variant, launch)
    rng, chunk = random.Random(launch.seed), 1 << 20
    # Python 3.10/3.12 getrandbits has a C-int bit-count limit. Word-aligned
    # chunks preserve the seeded byte stream without a single huge request.
    data = b"".join(rng.randbytes(min(chunk, sizes["input_bytes"] - start))
                    for start in range(0, sizes["input_bytes"], chunk))
    if variant.parameters["fine"]:
        data = bytearray(data)
        for index, nxt in enumerate(cycle(launch.count, launch.seed)):
            struct.pack_into("<I", data, index * variant.parameters["transfer_bytes"], nxt)
        data = bytes(data)
    return data, sizes


def make_checker(variant, launch, input_data):
    """Precompute this immutable input/launch's oracle once, outside GPU timing.

    Throughput checksums are circular XOR windows. Whole even cycles cancel;
    adjacent CTAs differ by only the departing and arriving first words. This
    avoids both reconstructing every tile and replaying every CTA's transfers.
    Full final tiles are still compared byte-for-byte for every sample.
    """
    sizes = validate_launch(variant, launch)
    if len(input_data) != sizes["input_bytes"]:
        raise ValueError("Input length mismatch")
    p = variant.parameters
    count, repetitions = launch.count, launch.repetitions
    stride = p["shape"][0] * p["element_bytes"] if p["shape"] else p["transfer_bytes"]

    def first_word(index):
        return struct.unpack_from("<I", input_data, index * stride)[0]

    value, index, final_index = 0, 0, 0
    if p["fine"]:
        for _ in range(repetitions):
            final_index = index
            index = first_word(index)
            if not 0 <= index < count:
                raise ValueError("TMA pointer escaped the input buffer")
        value = index
    else:
        full_cycles, remainder = divmod(repetitions, count)
        # Odd full cycles XOR the prefix a second time, leaving its complement.
        indices = range(remainder, count) if full_cycles % 2 else range(remainder)
        for index in indices:
            value ^= first_word(index)

    expected, final_tiles = [], {}
    for block in range(launch.blocks):
        if not p["fine"]:
            if block and remainder:
                value ^= first_word((block - 1) % count) ^ first_word((block - 1 + remainder) % count)
            final_index = (repetitions - 1 + block) % count
        if final_index not in final_tiles:
            final_tiles[final_index] = bytes(tma_tile(input_data, p, count, final_index))
        expected.append((value, final_tiles[final_index]))
    expected = tuple(expected)
    record_bytes = 32 + p["transfer_bytes"]

    def check_output(output):
        if len(output) != sizes["output_bytes"]:
            raise ValueError("Output length mismatch")
        cycles = []
        for block, (value, tile) in enumerate(expected):
            offset = block * record_bytes
            actual_cycles, actual_value, status = struct.unpack_from("<QII", output, offset)
            if actual_value != value or status:
                raise ValueError("TMA checksum/status mismatch")
            if output[offset + 32:offset + record_bytes] != tile:
                raise ValueError("TMA complete last-tile mismatch")
            if actual_cycles <= 0:
                raise ValueError("Missing/nonpositive cycle sample")
            cycles.append(actual_cycles)
        return {"passed": True, "cycles": cycles,
                "scope": "complete final tile and chain/checksum; not every transferred tile"}

    return check_output


def check(variant, launch, output, input_data):
    """One-shot API; runners reuse make_checker for repeated timing samples."""
    return make_checker(variant, launch, input_data)(output)
