"""Ordinary-load latency plan matching the existing TMA working-set points."""
import argparse
from dataclasses import asdict
import json

from .catalog import variants
from .inputs import Launch, validate_launch


def plan(variant, sm_count, *, l2_bytes, global_bytes=512 * 1024 * 1024, seed=17):
    if sm_count <= 0 or l2_bytes <= 0 or global_bytes <= l2_bytes:
        raise ValueError("Measured/device geometry required")
    jobs = []
    # Match TMA's 16-byte nodes capped at 2**24 nodes (256 MiB), even though
    # this reference uses 32-byte node spacing. Never assume a cache capacity.
    limit = min(global_bytes, 256 * 1024 * 1024)
    for exponent in range(8, limit.bit_length()):
        size = 1 << exponent
        if size > limit:
            continue
        launch = Launch(count=size // variant.parameters["stride"], repetitions=1048576,
                        seed=seed, memory_limit=global_bytes * 2)
        jobs.append({"variant": variant.id, "launch": asdict(launch),
                     "memory": validate_launch(variant, launch), "status": "not_run"})
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=variants()[0].id, choices=[v.id for v in variants()])
    parser.add_argument("--sm-count", required=True, type=int)
    parser.add_argument("--l2-bytes", required=True, type=int)
    parser.add_argument("--global-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(plan(variants()[0], args.sm_count, l2_bytes=args.l2_bytes,
                          global_bytes=args.global_bytes), indent=2))
