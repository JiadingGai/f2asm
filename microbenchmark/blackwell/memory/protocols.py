"""Retained-work checks for the pinned ordinary-load kernel, not timing proof."""
from ..common.sass import cfg, opcode


def inspect(variant, ptx, rows):
    clocks = [r["pc"] for r in rows if "SR_CLOCKLO" in r["sass"]]
    loads = [r for r in rows if opcode(r).startswith("LDG")]
    timed = [r for r in loads if len(clocks) == 2 and clocks[0] < r["pc"] < clocks[1]]
    conditions = [
        ("exact PTX target", ".target sm_100a" in ptx),
        ("single-thread ABI", ".reqntid 1, 1, 1" in ptx),
        ("L1-enabled compiler seed", "ld.global.ca.u32" in ptx),
        ("two clock reads", len(clocks) == 2),
        ("retained dependent measured loads", len(timed) == 8),
        ("retained untimed warmup loads", len(loads) == 16 and len(timed) == 8),
        ("compiler-verified scalar cache-enabled loads",
         bool(loads) and all(opcode(r) == "LDG.E.STRONG.SM" for r in loads)),
        ("observable final index", any("STG.E desc[UR4][R2.64+0x8], R15;" == r["sass"] for r in rows)),
        ("no TMA/shared-memory accesses", not any(opcode(r).startswith(("UBLKCP", "UTMA", "LDS", "SYNCS")) for r in rows)),
        ("resolved control flow", cfg(rows)["complete"]),
        ("full-chain warmup in seed", "warm_loop:" in ptx),
    ]
    return {"checks": [{"check": name, "passed": bool(ok)} for name, ok in conditions],
            "errors": [name for name, ok in conditions if not ok],
            "proof_scope": "pinned code structure; CPU tests additionally interpret integer/data flow",
            "limitations": ["Clock serialization and cache residency require hardware qualification",
                            "Final-index correctness is not a dynamic trace of every GPU load"]}
