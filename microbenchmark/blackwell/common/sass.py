"""Conservative static SASS inspection; unknown scheduling semantics stay unknown.

CFG edges express control flow ONLY. Scoreboard/stall/reuse annotations do not
invent edges or split a block at every predicated arithmetic instruction.
"""
from collections import Counter
import json
import re


def normalized(s): return re.sub(r"\s+", "", s).rstrip(";")


def records(raw, words, text=None):
    doc = json.loads(raw)
    if len(doc) != 2 or not doc[1]: raise ValueError("Expected kernel disassembly")
    fs=doc[1]; f=fs[0]
    if f["function-name"] != "bench" or f["start"] != 0 or f["length"]!=len(words)*16: raise ValueError("Wrong function extent")
    symbols={x["function-name"]:x["start"] for x in fs}
    slots={}
    for function in fs:
        for i,x in enumerate(function["sass-instructions"]):
            pc=function["start"]+16*i
            if pc in slots or pc%16 or not 0<=pc<len(words)*16: raise ValueError("Overlapping/out-of-bounds subfunction")
            slots[pc]=x
    if set(slots)!=set(range(0,len(words)*16,16)): raise ValueError("Incomplete instruction coverage")
    text_slots={}
    if text is not None:
        # nvdisasm 13.4 JSON omits separators in LDS/SYNCS/UBLKCP operands.
        # Use its independent text syntax, authenticated by BOTH printed halves
        # of the instruction word. These words are checks, never encoder output.
        lines=text.splitlines()
        for i,line in enumerate(lines[:-1]):
            m=re.search(r"/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;\s*/\* 0x([0-9a-fA-F]{16}) \*/",line)
            if not m: continue
            hi=re.search(r"/\* 0x([0-9a-fA-F]{16}) \*/",lines[i+1])
            pc=int(m[1],16)
            if not hi or pc in text_slots or pc%16 or not 0<=pc<len(words)*16: raise ValueError("Invalid text disassembly slot")
            if int(m[3],16)|(int(hi[1],16)<<64)!=words[pc//16]: raise ValueError("Printed instruction word differs from CUBIN")
            text_slots[pc]=m[2]+";"
        if set(text_slots)!=set(slots): raise ValueError("Text/JSON PC coverage differs")
    rows = []
    for i,w in enumerate(words):
        x=slots[i*16]
        if not x.get("opcode"): raise ValueError("Unrecognized disassembly slot")
        operands=x.get("operands","")
        # Only same-section, explicitly named disassembler functions. Full
        # F2Asm re-encoding must independently agree; this is not a linker.
        if x["opcode"].split(".")[0] in ("CALL","RET"):
            pieces=operands.split(",")
            pieces=[hex(symbols[t.strip()]) if t.strip() in symbols else t for t in pieces]
            operands=",".join(pieces)
        s = " ".join(filter(None,(x.get("predicate"),x["opcode"],operands)))+";"
        if text is not None and x["opcode"].split(".")[0] not in ("CALL","RET","BRA","BSSY"):
            textual=text_slots[i*16]
            prefix=" ".join(filter(None,(x.get("predicate"),x["opcode"])))
            if not textual.startswith(prefix): raise ValueError("Text/JSON opcode or predicate disagrees")
            s=textual
        rows.append({"pc":i*16,"sass":s,"control":(w>>105)&0x1ffff})
    return rows


def emit_source(rows):
    lines = ["# Explicit instruction/control source; compiler-derived ABI and benchmark template.",
             "# F2Asm is required to encode every instruction. No copied word fallback.",
             "# Static encoding qualification is NOT hardware semantic/timing validation."]
    for r in rows:
        lines += [f"pc_{r['pc']:04x}:", f"{r['control']:#07x} | {r['sass']}"]
    return "\n".join(lines)+"\n"


def parse_source(text):
    rows, labels = [], {}
    for lineno, raw in enumerate(text.splitlines(),1):
        s = raw.split("#",1)[0].strip()
        if not s: continue
        if s.endswith(":"):
            label = s[:-1]
            if not re.fullmatch(r"[A-Za-z_]\w*",label) or label in labels: raise ValueError("Invalid/duplicate label")
            labels[label] = len(rows)*16; continue
        control, sep, ins = s.partition("|")
        if not sep or not ins.strip().endswith(";"): raise ValueError(f"Bad SASS source line {lineno}")
        control = int(control,0)
        if not 0 <= control < 1<<17: raise ValueError("Control outside 17-bit field")
        rows.append({"pc":len(rows)*16,"control":control,"sass":ins.strip()})
    for r in rows:
        m = re.fullmatch(r"((?:@!?(?:UP|P)\d+\s+)?BRA(?:\.\w+)*\s+(?:!?(?:UP|P)(?:\d+|T)\s*,\s*)?)([A-Za-z_]\w*);",r["sass"])
        if m:
            if m[2] not in labels or labels[m[2]] >= len(rows)*16: raise ValueError("Unresolved branch")
            r["sass"] = m[1]+hex(labels[m[2]])+";"
    if not rows: raise ValueError("Empty SASS")
    return rows


def opcode(row):
    return re.sub(r"^@\S+\s+","",row["sass"]).split()[0].rstrip(";")


def controls(raw):
    if not 0 <= raw < 1<<17: raise ValueError("Invalid control")
    return {"stall":raw&15,"yield_bit":(raw>>4)&1,"write_slot":(raw>>5)&7,
            "read_slot":(raw>>8)&7,"wait_mask":(raw>>11)&63}


def cfg(rows):
    pcs = {r["pc"] for r in rows}
    if [r["pc"] for r in rows] != list(range(0,len(rows)*16,16)): raise ValueError("Noncontiguous instructions")
    leaders, targets, errors, unknown = {0}, {}, [], []
    for r in rows:
        op, pc = opcode(r).split(".")[0], r["pc"]
        if op in ("BRA","CALL"):
            m = re.search(r"\b(?:BRA|CALL)(?:\.\w+)*\s+(?:!?(?:UP|P)(?:\d+|T)\s*,\s*)?(0x[0-9a-fA-F]+);$",r["sass"])
            if not m: errors.append(f"Unresolved direct branch at {pc:#x}"); continue
            dest = int(m[1],16)
            if dest not in pcs: errors.append(f"Out-of-text branch at {pc:#x}"); continue
            leaders.add(dest); targets[pc] = dest
        elif op in ("BRX","JMP","JMX","RET","BSSY","BSYNC"):
            unknown.append(f"Control instruction {op} at {pc:#x} requires additional interpretation")
        if op in ("BRA","CALL","EXIT","TRAP","RET","JMP","BRX","JMX") and pc+16 in pcs: leaders.add(pc+16)
    boundaries = sorted(leaders)+[len(rows)*16]
    blocks=[]
    for lo,hi in zip(boundaries,boundaries[1:]):
        last=rows[hi//16-1]; op=opcode(last).split(".")[0]
        pred=last["sass"].startswith("@") or bool(re.search(r"\bBRA(?:\.\w+)*\s+!?(?:UP|P)\d+\s*,",last["sass"]))
        succ=[]
        if last["pc"] in targets: succ.append(targets[last["pc"]])
        if hi in pcs and (op not in ("BRA","EXIT","TRAP","RET","JMP","BRX","JMX") or pred): succ.append(hi)
        blocks.append({"start":lo,"end_exclusive":hi,"instructions":(hi-lo)//16,"successors":sorted(set(succ))})
    reachable=set(); todo=[0]; lookup={b["start"]:b for b in blocks}
    while todo:
        b=todo.pop()
        if b in reachable: continue
        reachable.add(b); todo.extend(lookup[b]["successors"])
    for b in blocks: b["reachable"]=b["start"] in reachable
    return {"blocks":blocks,"errors":errors,"unknowns":unknown,"complete":not errors and not unknown}


def audit(rows, family):
    graph=cfg(rows)
    opcounts=Counter(opcode(r).split(".")[0] for r in rows)
    clocks=[r["pc"] for r in rows if "SR_CLOCKLO" in r["sass"]]
    unknowns=list(graph["unknowns"])+[
        "CS2R timing/issue serialization is not established by static analysis",
        "Scoreboard slot reuse, fixed-latency stalls and predication require hardware qualification",
        "Cache residency and achieved throughput cannot be inferred from cache modifiers",
        "Blackwell auxiliary binary representation/loaded-code identity requires on-target verification"]
    errors=list(graph["errors"])
    if len(clocks)<2: errors.append("No complete pair of clock reads")
    if family=="tma": unknowns += ["TMA descriptor and asynchronous completion protocol need GPU validation"]
    return {"cfg":graph,"opcode_counts":dict(opcounts),"clock_pcs":clocks,
            "controls":[{"pc":r["pc"],**controls(r["control"])} for r in rows],
            "errors":errors,"unknowns":unknowns,"hardware_semantics_proven":False}


def dot(graph):
    out=["digraph cfg {",'node [shape=box,fontname="monospace"];']
    for b in graph["blocks"]:
        out.append(f'B{b["start"]} [label="{b["start"]:#x}..{b["end_exclusive"]:#x}\\n{b["instructions"]} instructions",style="{"solid" if b["reachable"] else "dashed"}"];')
        for d in b["successors"]: out.append(f'B{b["start"]} -> B{d};')
    return "\n".join(out+["}"])+"\n"
