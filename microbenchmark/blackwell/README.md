# Blackwell microbenchmark examples

This directory contains selected experimental examples extracted from the local
`blackwell-microbench` research repository. Families currently include
[TMA](tma/README.md) and its [ordinary-load latency reference](memory/README.md).
It is a source-checkout example, not an additional dependency of the F2Asm package.
The assembler implementation in `src/f2asm/` is unchanged.

## Run the complete TMA campaign

See [RUNBOOK.md](RUNBOOK.md) for the Runpod deployment, commands, failure recovery,
result retrieval, and the first B200 session's execution record.
The [B200 baseline results](results/b200-20261011/README.md) include all 1,080
samples, Figures 2–4 and commands to verify/replot them without a GPU.

`python3 -m microbenchmark.blackwell.campaign` is the run-all entry point.
The `tma` suite includes its ordinary-load latency reference. With default buffer
limits it runs **216 measurement configurations**: 21 TMA latency,
21 ordinary-load latency, and 29 TMA throughput
variants at S, 2S, 3S, 4S, 5S and 6S blocks. Each measurement has five samples by
default. There is **no separate 31-variant smoke phase**: each configuration's
existing worker already performs a short correctness proof before timing and
checks every timed sample.

The first full B200 campaign passed **216/216 configurations and 1,080 samples**
in 13 minutes 40 seconds. These remain experimental protocols, not
publication-qualified timings. Gather4, CTA-pair and new swizzle
modes are deferred. Future suites can extend the registry in `campaign.py` while
reusing execution, checkpointing and reporting.

CPU tests cover planning, per-configuration correctness, stop-on-failure,
checkpoint/retry handling, report integrity and completed-job resume. Synthetic
outputs test the runner only; they are never hardware measurements.
The measurement-only runner passes 81 CPU tests on Python 3.10/3.12/3.14. Its
full synthetic campaign completed 216 jobs and 1,080 samples; resume launched
zero additional workers. Completed-campaign resume also launched zero additional
workers on B200. See the runbook for the initial Python 3.12 large-buffer failure,
its input-generation fix, and the separately preserved successful rerun.

### 1. Preview locally without a GPU

From the F2Asm checkout root, use example geometry only to inspect the plan:

```sh
python3 -m microbenchmark.blackwell.campaign --suite tma --dry-run \
  --sm-count 128 --l2-bytes 67108864 \
  --output microbenchmark/blackwell/artifacts/tma-plan
```

This never loads CUDA, compiles, assembles, or launches a kernel. It writes the
manifest and empty, explicitly incomplete reports/plots. Use a **different** output
directory for real measurements; an offline plan cannot be resumed as GPU evidence.

### 2. Prepare the B200 host

- Use an isolated Linux B200 with compatible CUDA drivers, `nvidia-smi`, Python
  3.10+, and this source tree. No Python plotting package or runtime assembler is
  required. The runner queries the real GPU's SM count and L2 size.
- Prepare both [TMA](tma/README.md#2-build-and-assemble-offline) and
  [ordinary-load](memory/README.md#2-compile-and-assemble-offline) qualifications.
  Copy their complete directories, including SASS and qualification reports.
- Allow at least **8 GiB available host RAM** for Python input/chain construction;
  the buffer budget is not a host-RSS limit. The default GPU input budget is
  512 MiB, with latency working sets through 256 MiB. Free GPU memory is checked.
- Do not run training or another benchmark on the device concurrently. The host
  watchdog can kill a worker but **cannot guarantee recovery from a GPU hang**.

### 3. Run everything with one command

```sh
python3 -m microbenchmark.blackwell.campaign --suite tma \
  --qualified tma=microbenchmark/blackwell/artifacts/tma-qualified \
  --qualified memory=microbenchmark/blackwell/artifacts/memory-qualified \
  --output microbenchmark/blackwell/artifacts/tma-b200-campaign \
  --device 0 --samples 5 --max-seconds 60 --worker-timeout 300 \
  --execute --accept-unvalidated-hardware-protocols
```

Execution is sequential: one worker/one configuration at a time. All prepared
binaries are authenticated before launch. Every measurement checks its proof and
all samples. Any failure, timeout, partial sample count or interruption stops the
campaign; later cases remain pending. `--max-seconds` bounds the sampling phase;
`--worker-timeout` includes CPU input/oracle setup, CUDA setup, sampling and cleanup.

### 4. Resume without overwriting evidence

Repeat the **same command and options** with `--resume`. Only complete, hash-checked
results are skipped. Sources, artifact hashes, GPU UUID/host/driver, Python version,
and plan/settings must still match. Changed inputs require a fresh campaign.
Never remove or edit a saved result to force a retry.

For a failed/interrupted attempt, first inspect `stderr.txt`, `run.json`, and the
GPU; recover/reset the isolated device as needed. Then explicitly add:

```text
--resume --retry-failed
```

`--retry-failed` confirms you have inspected/recovered the GPU; it does not reset
the GPU. Retries create a new attempt directory and
preserve previous evidence. An interrupted `running` checkpoint is treated as
uncertain execution, even if its worker left a seemingly complete report. A file
lock prevents simultaneous writers to the same campaign directory.

### 5. Inspect results

- `manifest.json`: immutable plan, source/artifact hashes and device identity.
- `state.json`: atomic progress checkpoint, statuses and attempt history.
- `runs/<job>/attempt-NNN/`: worker logs, raw output, source snapshots and `run.json`.
- `summary.json`, `summary.csv`, `samples.csv`: all configurations/statuses and raw
  sample metrics. Only complete, verified passing cases contribute aggregates.
- `latency.svg`: median cycles/iteration versus working-set size for both curves.
- `throughput.svg`: requested decimal GB/s by transfer size and S–6S grid, with
  a separate equal-byte shape panel. Missing/failed cases are not plotted as zero.

Raw results and progress are saved per configuration. Aggregate reports/plots are
generated when the campaign completes or stops, not rebuilt after every job.

To regenerate reports later without a GPU:

```sh
python3 -m microbenchmark.blackwell.campaign --report-only \
  --output microbenchmark/blackwell/artifacts/tma-b200-campaign
```

Require `state.json` status `completed` and all 216 measurement rows `passed`.
Before stopping a rented GPU, run the same command with `--verify-only` instead:
it fails on incomplete campaigns or missing/changed raw evidence. `--report-only`
intentionally permits incomplete reports and is not a completion check.
Keep the entire directory. Requested transfer throughput is **not distinct HBM
bandwidth**; TMA and ordinary-load latency differ in node spacing and warm-up, so
subtracting their curves does not isolate TMA hardware overhead.

## Lightweight correctness checks

Both host runners compute expected results once per input/launch configuration,
before GPU timing, then reuse them for every sample. TMA throughput uses a rolling
XOR of first words instead of reconstructing every transfer. Each sample still
checks status, checksum/final index, positive cycle count, and (for TMA) every
byte of the final tile. No correctness-disable flag is needed.

`run.json` records `cpu_oracle_setup_seconds` and each sample's
`cpu_check_seconds` separately from GPU timing. This reduces host overhead;
it does not change the SASS, GPU measurement protocol, or correctness coverage.
The static CFG analyzer treats `BPT.TRAP` as a terminating instruction, retaining
fallthrough only when predicated; other breakpoint forms remain explicit unknowns.

Run all CPU regressions from the checkout root:

```sh
python3 -m unittest discover -s microbenchmark/blackwell -t . -v
```

The checker/CFG revision passed 45 CPU tests on Python 3.10, 3.12 and 3.14.
Fresh offline assembly reproduced all 31 CUBINs byte-for-byte (10,728 instruction
words), with correct trap termination in all 30 TMA CFGs. Historical import and
validation records remain unchanged; these checks are not GPU validation.

## Incremental import rules

- Each family owns its runner, seed generator, SASS, protocol checks and tests.
  Future memory/tensor families belong in sibling directories, not inside `tma/`.
- `common/build.py` and `common/runner.py` provide the shared build and host
  workflow; family entry points supply protocol-specific hooks. Do not recopy
  build or CUDA-execution infrastructure when importing another family.
- `common/` owns only reusable CUDA, artifact, statistics and static-inspection
  helpers. Families import these helpers; do not duplicate or recopy them.
- `tma/IMPORT.json` records the original commit, original file hashes, extraction
  mappings and destination hashes at import. This is an immutable import baseline,
  not a claim that later edits must be forbidden.
- Before another import, run the drift check below. Review source changes and
  destination changes against the recorded baseline. If either side changed, use
  a three-way review; never overwrite shared helpers or refresh hashes just to
  silence a difference. New families get their own import record.
- Preserve the research repository and its artifacts. Build binaries, models and
  raw campaign archives stay in ignored artifact directories. Reviewed, de-identified
  result exports may be published separately under `results/`.
- Local import, hardware validation and upstream publication are separate steps.
  Nothing here establishes GPU correctness, accurate timings, or permission to push.

From the F2Asm checkout root:

```sh
python -m microbenchmark.blackwell.common.provenance \
  microbenchmark/blackwell/tma/IMPORT.json \
  --source-repo /path/to/blackwell-microbench
```

Omit `--source-repo` to check only destination drift. The checker is read-only and
returns nonzero when review is needed. Review shared-helper changes separately
from any benchmark protocol changes.
