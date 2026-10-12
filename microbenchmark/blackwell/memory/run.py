"""Ordinary-load reference entry point; prepared CUBIN execution only."""
from pathlib import Path
from types import SimpleNamespace

from ..common import runner
from ..common.telemetry import telemetry
from .catalog import variants
from .inputs import Launch, check, make_checker, prepare, validate_launch


def work(variant, launch):
    return {"dependent_loads": launch.repetitions}


def measurement_metadata(variant):
    return {"cycle_metric": "clock_delta_including_dependent_address_and_loop_overhead",
            "cycle_normalization": "divide cycles by repetitions for cycles per dependent load",
            "event_metric": "whole_kernel_including_full_chain_warmup_and_result_export",
            "traffic_metric": "dependent loads, not HBM transactions or bandwidth",
            "load_bytes": 4, "node_stride_bytes": variant.parameters["stride"],
            "cache_policy": "compiler seed ld.global.ca; residency requires GPU validation",
            "warmup": "one complete chain traversal before the clock interval",
            "comparison_limit": "TMA uses 16-byte nodes and a different warmup/completion protocol"}


def suite():
    return SimpleNamespace(directory=Path(__file__).resolve().parent, variants=variants,
                           Launch=Launch, prepare=prepare, check=check, make_checker=make_checker,
                           validate_launch=validate_launch,
                           work=work, measurement_metadata=measurement_metadata, telemetry=telemetry)


def execute(args, variant, launch, directory, driver_factory=None):
    return runner.execute(args, variant, launch, directory, suite(), driver_factory)


def main():
    runner.main(suite())


if __name__ == "__main__":
    main()
