"""TMA runtime plans. Requested bytes are not a claim about distinct HBM traffic."""
import argparse
from dataclasses import asdict
import json

from .catalog import variants
from .inputs import Launch, validate_launch


def plan(variant, sm_count, *, l2_bytes, global_bytes=512 * 1024 * 1024, seed=17):
    if sm_count <= 0 or l2_bytes <= 0 or global_bytes <= l2_bytes:
        raise ValueError("Measured/device geometry required")
    p, jobs = variant.parameters, []
    def add(**kwargs):
        launch = Launch(seed=seed, memory_limit=global_bytes * 2, **kwargs)
        jobs.append({"variant": variant.id, "launch": asdict(launch),
                     "memory": validate_launch(variant, launch), "status": "not_run"})
    if p["fine"]:
        for exponent in range(8, global_bytes.bit_length()):
            size = 1 << exponent
            if size <= global_bytes and size // p["transfer_bytes"] <= 1 << 24:
                add(count=size // p["transfer_bytes"], repetitions=1048576)
    else:
        for multiplier in (1, 2, 3, 4):
            blocks = sm_count * multiplier
            reps = max(1, (4_000_000_000 + blocks * p["transfer_bytes"] - 1)
                       // (blocks * p["transfer_bytes"]))
            add(blocks=blocks, count=min(1 << 24, global_bytes // p["transfer_bytes"]),
                repetitions=min(reps, 1 << 24))
            jobs[-1]["volume_convention"] = "decimal 4 GB requested; overlapping CTA streams retained"
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=[v.id for v in variants()])
    parser.add_argument("--sm-count", required=True, type=int)
    parser.add_argument("--l2-bytes", required=True, type=int)
    parser.add_argument("--global-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    variant = next(v for v in variants() if v.id == args.variant)
    print(json.dumps(plan(variant, args.sm_count, l2_bytes=args.l2_bytes,
                          global_bytes=args.global_bytes), indent=2))
