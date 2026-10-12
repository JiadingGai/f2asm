"""TMA host entry point. Dry-run by default; no runtime assembly or training."""
from pathlib import Path
from types import SimpleNamespace

from ..common import runner
from ..common.telemetry import telemetry
from .catalog import variants
from .inputs import Launch, check, make_checker, prepare, validate_launch


def work(variant, launch):
    return {"transferred_bytes": launch.blocks * launch.repetitions * variant.parameters["transfer_bytes"]}


def measurement_metadata(variant):
    return {"cycle_metric": "clock_delta_including_loop_and_completion_overhead",
            "event_metric": "whole_kernel_including_setup_and_correctness_export",
            "traffic_metric": "requested transfer bytes, not distinct HBM traffic"}


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
