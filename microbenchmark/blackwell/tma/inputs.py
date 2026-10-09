"""TMA-only inputs, launch bounds and final-tile/checksum oracles; no CUDA dependency."""
from dataclasses import dataclass
import random
import struct


@dataclass(frozen=True)
class Launch:
    blocks: int = 1
    repetitions: int = 1024
    count: int = 1024
    seed: int = 0
    memory_limit: int = 512 * 1024 * 1024


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


def cycle(nodes, seed=0):
    if type(nodes) is not int or not 1 <= nodes <= 1 << 24:
        raise ValueError("Invalid node count")
    order = list(range(nodes))
    random.Random(seed).shuffle(order)
    nxt = [0] * nodes
    for a, b in zip(order, order[1:] + order[:1]):
        nxt[a] = b
    return nxt


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
    data = random.Random(launch.seed).randbytes(sizes["input_bytes"])
    if variant.parameters["fine"]:
        data = bytearray(data)
        for index, nxt in enumerate(cycle(launch.count, launch.seed)):
            struct.pack_into("<I", data, index * variant.parameters["transfer_bytes"], nxt)
        data = bytes(data)
    return data, sizes


def check(variant, launch, output, input_data):
    sizes = validate_launch(variant, launch)
    if len(output) != sizes["output_bytes"] or len(input_data) != sizes["input_bytes"]:
        raise ValueError("Input/output length mismatch")
    p = variant.parameters
    cycles = []
    for block in range(launch.blocks):
        offset = block * (32 + p["transfer_bytes"])
        value, index = 0, 0
        for iteration in range(launch.repetitions):
            tile = tma_tile(input_data, p, launch.count,
                            index if p["fine"] else (iteration + block) % launch.count)
            first = struct.unpack_from("<I", tile)[0]
            if p["fine"]:
                value, index = first, first
            else:
                value ^= first
        actual_cycles, actual_value, status = struct.unpack_from("<QII", output, offset)
        if actual_value != value or status:
            raise ValueError("TMA checksum/status mismatch")
        if output[offset + 32:offset + 32 + len(tile)] != tile:
            raise ValueError("TMA complete last-tile mismatch")
        if actual_cycles <= 0:
            raise ValueError("Missing/nonpositive cycle sample")
        cycles.append(actual_cycles)
    return {"passed": True, "cycles": cycles,
            "scope": "complete final tile and chain/checksum; not every transferred tile"}
