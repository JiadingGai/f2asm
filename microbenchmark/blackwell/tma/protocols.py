"""TMA structural checks, not a proof of hardware semantics or timing accuracy."""
from ..common.sass import opcode


def inspect(variant, ptx, rows):
    p = variant.parameters
    ops = [opcode(row) for row in rows]
    has = lambda prefix: any(op.startswith(prefix) for op in ops)
    conditions = [
        ("exact PTX target", ".target sm_100a" in ptx),
        ("two retained clock reads", sum("SR_CLOCKLO" in r["sass"] for r in rows) >= 2),
        ("observable global result", has("STG")),
        ("TMA issue retained", has("UTMALDG") if p["shape"] else has("UBLKCP")),
        ("barrier initialization retained", has("SYNCS.EXCH")),
        ("expected transaction bytes retained", has("SYNCS.ARRIVE.TRANS64")),
        ("completion phase wait retained", has("SYNCS.PHASECHK")),
        ("declared expected byte count", f"[%r12],{p['transfer_bytes']};" in ptx),
        ("full final-tile correctness export", "export_loop:" in ptx),
        ("bounded wait with failure trap", "1000000" in ptx and "trap;" in ptx),
    ]
    return {"checks": [{"check": name, "passed": bool(ok)} for name, ok in conditions],
            "errors": [name for name, ok in conditions if not ok],
            "proof_scope": "protocol skeleton and retained work only"}
