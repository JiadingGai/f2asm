# B200 ordinary-load latency reference

`memory_l1_pointer_chase` supplies the non-TMA curve for a Blackwell version of
Figure 2 in [the Hopper microbenchmarking paper](https://arxiv.org/abs/2501.12084).
It measures a randomized dependent chain using ordinary global loads, alongside
the existing [TMA completion-latency curve](../tma/README.md).

The checked-in SASS and PTX seed are extracted unchanged from
`blackwell-microbench` commit `5d910e01d32d2c2faa87475e650d0ee96e3f12b9`.
Shared compilation, artifact authentication, CUDA execution and statistics live
in `../common/`; this family has no runtime dependency on the research repo or TMA.
See [IMPORT.json](IMPORT.json) for the separate immutable import baseline.

## Measurement contract

- One CTA, one thread; 4-byte `LDG` accesses to 32-byte-spaced nodes. The compiler
  seed requests `ld.global.ca.u32` (L1 enabled), producing the pinned
  `LDG.E.STRONG.SM` sequence. This does not prove where any access actually hits.
- One complete chain traversal warms the working set before the cycle timer.
  The timed region performs exactly `repetitions` dependent loads; there is no
  checksum or output store in the measured loop. Address arithmetic and loop
  control remain included. The CPU computes the expected final index once per
  input/launch and checks every sample against it; it does not rewalk the chain
  after each sample. [Host checking costs](../README.md#lightweight-correctness-checks)
  are recorded separately from GPU timing.
- Default sweep: 21 working sets, powers of two from 256 B through 256 MiB,
  with 1,048,576 measured loads per point. These match the TMA sweep's working-set
  sizes, not its node counts. Larger-device studies may need an extended range.
- Use `raw_samples[].cycles_per_iteration` or
  `cycles_per_iteration_statistics` for cycles per dependent load. CUDA-event
  time includes the full-chain warm-up and is **not** the latency-curve metric.
- The TMA baseline uses 16-byte nodes, completion barriers and a different
  warm-up protocol. Plot the curves with these distinctions documented; their
  difference is **not** an isolated measurement of TMA hardware overhead.
- This is one Blackwell ordinary-load curve, not reruns on A100/RTX4090/Hopper.
  Gather4, CTA-pair, new swizzle modes and other memory benchmarks are not included.

The first B200 campaign passed all 21 ordinary-load configurations, with five
timed samples each; see the [session runbook](../RUNBOOK.md). This establishes
checked outputs for that run, not clock serialization, cache residency or timing
accuracy. [VALIDATION.json](VALIDATION.json) retains the historical CPU-only record.

## Runbook

Run from the F2Asm checkout root; Python 3.10+ is sufficient for CPU checks and
prepared-CUBIN execution. Use fresh output directories for every attempt.

### 1. Check and plan (no GPU)

```sh
python3 -m unittest discover -s microbenchmark/blackwell -t . -v
python3 -m microbenchmark.blackwell.memory.run \
  --variant memory_l1_pointer_chase --dry-run
python3 -m microbenchmark.blackwell.memory.sweeps \
  --sm-count "${SM_COUNT:?Read from the target GPU}" \
  --l2-bytes "${L2_BYTES:?Read from the target GPU}"
```

The planner prints JSON; it does not launch kernels. Combined with the current
174 TMA throughput configurations and 21 TMA latency points, this adds 21 ordinary
latency points: **216 default configurations**, before samples/correctness launches.
The [TMA campaign runner](../README.md#run-the-complete-tma-campaign) executes both
families, supports safe resume, and produces the combined latency plot.

### 2. Compile and assemble offline

On the Linux build host, use pinned PTXAS 12.8.93 and nvdisasm 13.4.92 (no GPU).
Offline assembly needs the full F2Asm implementation and the existing Blackwell
model/compiler supplement, supplied explicitly; the incremental public learning
core alone is insufficient. No training or downloading occurs implicitly.

```sh
python3 -m microbenchmark.blackwell.memory.build compile \
  --ptxas /path/to/ptxas --nvdisasm /path/to/nvdisasm \
  --output microbenchmark/blackwell/artifacts/memory-build
PYTHONPATH=/path/to/full-f2asm/src python3 -m microbenchmark.blackwell.memory.build assemble \
  --build microbenchmark/blackwell/artifacts/memory-build \
  --model /path/to/blackwell-encoder.json \
  --supplement /path/to/compiler-supplement.json \
  --output microbenchmark/blackwell/artifacts/memory-qualified
```

The original research `artifacts/build-v4` can also be supplied as `--build`.
Only this reference is selected. The manifest pins seed, SASS and CUBIN hashes;
all 112 instructions must be F2Asm-encoded and match, with no copied-word fallback.
The seven compiler-evidence instructions are not held-out inference.
Arbitrary SASS edits need a separately reviewed patch/resource contract.

### 3. Smoke-test and measure on an isolated B200

Copy the `microbenchmark/` source tree and qualified artifacts to the GPU host.
The host loads a prepared CUBIN; it does not invoke the compiler or assembler.

```sh
python3 -m microbenchmark.blackwell.memory.run \
  --variant memory_l1_pointer_chase \
  --qualified microbenchmark/blackwell/artifacts/memory-qualified \
  --output microbenchmark/blackwell/artifacts/memory-smoke \
  --count 7 --repetitions 9 --samples 1 --max-seconds 10 \
  --execute --accept-unvalidated-hardware-protocols
```

After a successful smoke test, generate the plan using `device.sm_count` and
`device.l2_bytes` from `run.json`. For each entry, pass its `launch.count`,
`repetitions`, `seed`, and `memory_limit` as `--count`, `--repetitions`, `--seed`
and `--memory-limit`; retain `--blocks 1`, use `--samples 5`, and a fresh output
directory. Run sequentially, stopping on errors. Keep `run.json`, raw outputs,
telemetry and source snapshots. Require every numerical check to pass; a host
watchdog timeout does not establish that the GPU recovered safely.
