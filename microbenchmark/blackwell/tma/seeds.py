"""Unchanged TMA PTX seed functions; execution uses offline F2Asm CUBINs."""


def entry(body, *, threads=1, shared=0, extra="", record_bytes=32):
    return f""".version 8.7
.target sm_100a
.address_size 64
.visible .entry bench(.param .u64 input, .param .u64 output,
 .param .u32 repetitions, .param .u32 count, .param .u64 tensor_map)
.reqntid {threads}, 1, 1
{{
 .reg .pred %p<16>;
 .reg .b32 %r<128>;
 .reg .b64 %d<32>;
 .reg .f32 %f<32>;
 {'.shared .align 128 .b8 smem['+str(shared)+'];' if shared else ''}
 {extra}
 ld.param.u64 %d0, [input]; ld.param.u64 %d1, [output];
 ld.param.u32 %r0, [repetitions]; ld.param.u32 %r1, [count];
 mov.u32 %r2, %tid.x; mov.u32 %r3, %ctaid.x;
 mad.lo.u32 %r4, %r3, {threads}, %r2;
 mul.wide.u32 %d2, %r4, {record_bytes}; add.u64 %d1, %d1, %d2;
 mov.u32 %r5, 0; mov.u32 %r6, 0;
 {body}
}}
"""



def tma(v):
    p = v.parameters
    size, shape = p["transfer_bytes"], p["shape"]
    if shape:
        coords = ",".join(["%r15"] + ["0"]*(len(shape)-1))
        issue = f"cp.async.bulk.tensor.{len(shape)}d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%r11],[%d13,{{{coords}}}],[%r12];"
        address = f"mul.lo.u32 %r15,%r13,{shape[0]};"
    else:
        issue = f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%r11],[%d3],{size},[%r12];"
        address = f"mul.wide.u32 %d3,%r13,{size}; add.u64 %d3,%d0,%d3;"
    return entry(f"""
 setp.ne.u32 %p0,%r2,0; @%p0 bra done;
 ld.param.u64 %d1,[output]; mul.wide.u32 %d2,%r3,{size+32}; add.u64 %d1,%d1,%d2;
 mov.u32 %r11,smem; mov.u32 %r12,bar;
 ld.param.u64 %d13,[tensor_map];
 mbarrier.init.shared.b64 [%r12],1; fence.proxy.async.shared::cta;
 mov.u32 %r10,0; mov.u32 %r14,0; mov.u32 %r13,0;
 mov.u64 %d4,%clock64;
 measure_loop:
 // count is the number of legal transfer tiles; round-robin over them.
 {'' if p['fine'] else 'add.u32 %r13,%r10,%r3; rem.u32 %r13,%r13,%r1;'} {address}
 mbarrier.arrive.expect_tx.shared.b64 %d14,[%r12],{size};
 {issue}
 mov.u32 %r16,0;
 wait_loop: mbarrier.try_wait.parity.shared.b64 %p1,[%r12],%r14;
 @%p1 bra complete; add.u32 %r16,%r16,1;
 setp.lt.u32 %p2,%r16,1000000; @%p2 bra wait_loop;
 mov.u32 %r5,1; st.global.u32 [%d1+12],%r5; trap;
 complete: ld.shared.u32 %r17,[%r11];
 {'mov.u32 %r13,%r17; mov.u32 %r6,%r17;' if p['fine'] else 'xor.b32 %r6,%r6,%r17;'}
 xor.b32 %r14,%r14,1;
 add.u32 %r10,%r10,1; setp.lt.u32 %p0,%r10,%r0; @%p0 bra measure_loop;
 mbarrier.inval.shared.b64 [%r12];
 report: mov.u64 %d5,%clock64; sub.u64 %d6,%d5,%d4;
 st.global.u64 [%d1],%d6; st.global.u32 [%d1+8],%r6; st.global.u32 [%d1+12],%r5;
 mov.u32 %r7,%smid; st.global.u32 [%d1+16],%r7;
 // Full last-tile export is outside the cycle-timed region for correctness.
 mov.u32 %r18,0;
 export_loop: add.u32 %r19,%r11,%r18; ld.shared.u32 %r20,[%r19];
 cvt.u64.u32 %d15,%r18; add.u64 %d15,%d1,%d15; st.global.u32 [%d15+32],%r20;
 add.u32 %r18,%r18,4; setp.lt.u32 %p3,%r18,{size}; @%p3 bra export_loop;
 done: ret;
 """, threads=32, shared=size, extra=".shared .align 8 .b64 bar;")



def generate(variant):
    if variant.family != "tma":
        raise ValueError("TMA extraction only")
    return tma(variant)
