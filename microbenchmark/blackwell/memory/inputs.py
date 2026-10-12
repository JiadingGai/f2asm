"""Bounded randomized dependent chains and a CPU final-index oracle."""
import struct

from ..common.chains import cycle
from ..common.types import Launch
from .catalog import variants


def validate_launch(variant, launch):
    if variant not in variants():
        raise ValueError("Unknown ordinary-load variant")
    for name in ("blocks", "repetitions", "count", "memory_limit"):
        value = getattr(launch, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if launch.blocks != 1:
        raise ValueError("Dependent latency requires one CTA")
    if launch.count > 1 << 24 or launch.repetitions > 1 << 24:
        raise ValueError("Bounded-batch limit exceeded")
    size = launch.count * variant.parameters["stride"]
    if size + 32 > launch.memory_limit:
        raise ValueError("Memory budget exceeded before allocation")
    return {"input_bytes": size, "output_bytes": 32, "threads": 1,
            "hardware_correctness": "NOT_RUN"}


def prepare(variant, launch):
    sizes = validate_launch(variant, launch)
    data = bytearray(sizes["input_bytes"])
    for index, nxt in enumerate(cycle(launch.count, launch.seed)):
        struct.pack_into("<I", data, index * variant.parameters["stride"], nxt)
    return bytes(data), sizes


def final_index(variant, launch, data):
    # The generated input is one closed permutation cycle. The untimed full
    # traversal returns to index zero. Walk the measured accesses exactly;
    # do not rely on a modulo shortcut that could hide a corrupt chain.
    index = 0
    for _ in range(launch.repetitions):
        if not 0 <= index < launch.count:
            raise ValueError("Pointer escaped the input buffer")
        index = struct.unpack_from("<I", data, index * variant.parameters["stride"])[0]
    if not 0 <= index < launch.count:
        raise ValueError("Pointer escaped the input buffer")
    return index


def make_checker(variant, launch, input_data):
    """Compute the chain oracle once per input/launch, not once per sample."""
    sizes = validate_launch(variant, launch)
    if len(input_data) != sizes["input_bytes"]:
        raise ValueError("Input length mismatch")
    expected = final_index(variant, launch, input_data)

    def check_output(output):
        if len(output) != sizes["output_bytes"]:
            raise ValueError("Output length mismatch")
        cycles, actual_index, status, sm = struct.unpack_from("<QIII", output)
        if actual_index != expected or status:
            raise ValueError("Final pointer/status mismatch")
        if cycles <= 0:
            raise ValueError("Missing/nonpositive cycle sample")
        return {"passed": True, "cycles": [cycles], "sm": sm,
                "scope": "final dependent-chain index and status, not an execution trace"}

    return check_output


def check(variant, launch, output, input_data):
    return make_checker(variant, launch, input_data)(output)
