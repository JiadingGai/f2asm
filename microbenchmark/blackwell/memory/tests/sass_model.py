"""Narrow sequential data-flow interpreter for the pinned kernel's timed path.

Not an ISA emulator or GPU simulator: no stalls, caches, scoreboard, or timing.
Unsupported instructions fail closed. Tests use it to detect DCE, wrong loop
bounds, broken dependencies and out-of-bounds accesses in the retained SASS.
"""
import re
import struct


def interpret(rows, data, count, repetitions):
    registers, predicates = {}, {"PT": True}
    reads, clocks, pc = [], [], 0
    constants = {0x37c: 0, 0x394: count, 0x390: repetitions,
                 0x380: 0, 0x384: 0, 0x388: 0x10000000, 0x38c: 0}

    def value(token):
        token = token.removesuffix(".reuse")
        if token.startswith("-"):
            return -value(token[1:])
        if token == "RZ":
            return 0
        return registers[token] if token.startswith("R") else int(token, 0)

    def wide(token):
        if token == "RZ":
            return 0
        index = int(token[1:])
        return value(token) | (value(f"R{index + 1}") << 32)

    def put(token, number, bits=32):
        registers[token] = number & 0xffffffff
        if bits == 64:
            registers[f"R{int(token[1:]) + 1}"] = (number >> 32) & 0xffffffff

    # Enough for warmup and measurement plus unrolled prologues/tails.
    for _ in range(100 + 25 * (count + repetitions)):
        row = rows[pc // 16]
        ins = row["sass"].rstrip(";")
        next_pc = pc + 16
        if ins.startswith("@"):
            guard, ins = ins.split(" ", 1)
            name = guard.lstrip("@!")
            active = predicates[name]
            if guard.startswith("@!"):
                active = not active
            if not active:
                pc = next_pc
                continue
        op, _, operands = ins.partition(" ")
        args = operands.split(", ")
        if op.startswith("LDCU"):
            pass  # Descriptor setup; no address arithmetic in this model.
        elif op.startswith("LDC"):
            offset = int(re.search(r"\[0x0\]\[(0x[0-9a-f]+)\]", args[1])[1], 16)
            put(args[0], constants[offset], 64 if ".64" in op else 32)
        elif op == "S2R":
            if args[1] != "SR_CTAID.X":
                raise ValueError("Unexpected special register")
            put(args[0], 0)
        elif op.startswith("IMAD"):
            dest, a, b, c = args
            put(dest, (value(a) & 0xffffffff) * (value(b) & 0xffffffff)
                + (wide(c) if ".WIDE" in op else value(c)), 64 if ".WIDE" in op else 32)
        elif op in ("IADD3", "VIADD"):
            put(args[0], sum(value(x) for x in (args[3:] if op == "IADD3" else args[1:])))
        elif op == "MOV":
            put(args[0], value(args[1]))
        elif op.startswith("ISETP"):
            comparison, combine = op.split(".")[1], op.split(".")[-1]
            a, b = value(args[2]) & 0xffffffff, value(args[3]) & 0xffffffff
            condition = {"GT": a > b, "GE": a >= b, "LE": a <= b, "LT": a < b}[comparison]
            other = predicates[args[4]]
            predicates[args[0]] = condition and other if combine == "AND" else condition or other
        elif op == "PLOP3.LUT" and args == ["P0", "PT", "PT", "PT", "PT", "0x8", "0x80"]:
            predicates["P0"] = False
        elif op == "LDG.E.STRONG.SM":
            address_reg = re.fullmatch(r"desc\[UR4\]\[(R\d+)\.64\]", args[1])[1]
            address = wide(address_reg)
            if address % 32 or not 0 <= address <= len(data) - 4:
                raise ValueError("Out-of-bounds or misaligned node load")
            reads.append(address // 32)
            put(args[0], struct.unpack_from("<I", data, address)[0])
        elif op == "BRA":
            next_pc = int(args[0], 0)
        elif op == "CS2R" and args[1] == "SR_CLOCKLO":
            clocks.append(len(reads))
            if len(clocks) == 2:
                return {"clock_load_counts": clocks, "final_index": value("R15"), "indices": reads}
            put(args[0], 0, 64)  # No timing is simulated.
        else:
            raise ValueError(f"Unsupported data-flow instruction: {ins}")
        pc = next_pc
        if not 0 <= pc < len(rows) * 16:
            raise ValueError("Out-of-bounds control flow")
    raise ValueError("Interpreter step bound exceeded")
