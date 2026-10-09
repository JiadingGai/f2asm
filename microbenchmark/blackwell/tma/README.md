# Experimental B200 TMA benchmarks

TMA-only extraction from `blackwell-microbench` commit
`5d910e01d32d2c2faa87475e650d0ee96e3f12b9`. The 30 explicit SASS files are copied
byte-for-byte; seed PTX, launch ABI, tile indexing and measurement bodies retain
the original behavior. No other benchmark family or assembler implementation is
copied. Shared infrastructure lives in `../common/` and is imported with qualified
package names, so later families cannot shadow generic `inputs`/`catalog` modules.

| Experiment | Variants |
|---|---:|
| Dependent 16-byte completion-latency chain | 1 |
| Bulk throughput, 1/2/4/8/12/16 KiB | 6 |
| 1D tensor throughput, 0.5/0.75/1/1.25/1.5/2 KiB | 6 |
| 2D tensor throughput, 1 through 16 KiB | 6 |
| 3D tensor throughput, 1 through 16 KiB | 6 |
| Equal-byte shape comparison, 16 KiB | 5 |

Scope follows Sections 5.1 and 5.2 of [Dissecting the NVIDIA Hopper Architecture
through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084),
adapted to B200 / `sm_100a`. These are compiler-derived SASS baselines, not newly
hand-written instruction bodies. Current qualification intentionally rejects
arbitrary SASS changes until their metadata/resource/scheduling contract is reviewed.

## Validation limits retained by this import

- All 30 original artifacts were statically reconstructed. Their 10,616 instruction
  words include 137 occurrences encoded by a separate compiler-evidence supplement.
  This is reconstruction with compiler evidence, not held-out prediction.
- Hardware correctness is **NOT_RUN**; performance is **NOT_MEASURED**. CPU tests,
  encoding equality and retained protocol instructions do not establish either.
- Throughput CTAs use `(iteration + block) % count`, so streams overlap. Requested
  transfer bytes, including the planned decimal 4 GB campaign, are not distinct HBM
  traffic. Do not label these results peak HBM bandwidth.
- One transfer completes before the next starts within each CTA. Address/checksum
  work is inside the cycle interval; CUDA-event timing also includes setup and final
  tile export. Clock serialization, cache residency and async semantics need B200 tests.
- The checker verifies every byte of the final tile plus a chain/checksum result,
  not every byte of every transfer. Full paper-style plots/reporting are not included.
- This extraction does not fix those protocols. Keep measurement fixes separate
  from the import so comparisons remain auditable.

## Runbook

Run commands from the F2Asm checkout root with Python 3.10+. Use fresh output
directories for each attempt; never overwrite prior evidence. Paths below are
repository-relative or placeholders for explicitly supplied tools/models.

### 1. Check locally (no GPU)

The extraction was checked on Python 3.12.14 and 3.14.7: 17 new CPU tests and
159 existing F2Asm tests passed. All 30 CUBINs were reassembled byte-identically,
and 90 input/oracle cases plus every runtime sweep plan matched the source repo.
See [the extraction validation record](VALIDATION.json). The independent compiler
seeds were reused and authenticated; this was not a fresh compiler or GPU run.

```sh
python3 -m unittest discover -s microbenchmark/blackwell/tma/tests -t . -v
python3 -m microbenchmark.blackwell.tma.catalog
python3 -m microbenchmark.blackwell.tma.run \
  --variant tma_tensor_3d_throughput--8x8x8 --dry-run
```

Only offline assembly requires F2Asm. Listing, dry-runs, CPU tests and prepared
CUBIN execution require the Python standard library, not F2Asm imports or a model.

### 2. Build and assemble offline

On the Linux build host, use PTXAS **12.8.93** and nvdisasm **13.4.92**.
No GPU is needed. Compilation and assembly run sequentially; replace the
`/path/to/...` placeholders before running.

```sh
python3 -m microbenchmark.blackwell.tma.build compile \
  --ptxas /path/to/ptxas --nvdisasm /path/to/nvdisasm \
  --output microbenchmark/blackwell/artifacts/tma-build

PYTHONPATH=src python3 -m microbenchmark.blackwell.tma.build assemble \
  --build microbenchmark/blackwell/artifacts/tma-build \
  --model /path/to/blackwell-encoder.json \
  --supplement /path/to/compiler-supplement.json \
  --output microbenchmark/blackwell/artifacts/tma-qualified
```

Check `build-report.json` and `qualification-report.json` in their respective
output directories: all 30 variants must pass before a full GPU campaign.

The existing research `artifacts/build-v4` directory may also be supplied via
`--build`; only the 30 TMA entries are selected. `--variant ID` selects one entry
for either step. The checked-in manifest pins the original PTX, SASS, CUBIN and
model hashes. Tool/binary drift fails closed and requires review.

Supply the Blackwell model and its existing compiler-evidence supplement explicitly.
The repository's Rubin model is not a substitute. Models are neither bundled nor
downloaded or trained implicitly. A more complete Blackwell model can omit the
supplement if it independently supports every word and reproduces the pinned CUBIN.
The source repository is not a runtime/build dependency once these artifacts and
models are provided. No user-specific path is embedded in these commands or code.

### 3. Smoke-test on an isolated B200

This step is **not yet hardware-tested**. Use Linux with compatible NVIDIA drivers.
Copy the `microbenchmark/` tree and qualified artifacts to the GPU host, retaining
their relative layout, and run from the checkout root. F2Asm assembly is **not**
called per run.

```sh
python3 -m microbenchmark.blackwell.tma.run \
  --variant tma_tensor_3d_throughput--8x8x8 \
  --qualified microbenchmark/blackwell/artifacts/tma-qualified \
  --output microbenchmark/blackwell/artifacts/tma-gpu-smoke \
  --device 0 --blocks 1 --count 3 --repetitions 2 --samples 1 --max-seconds 10 \
  --execute --accept-unvalidated-hardware-protocols
```

The host authenticates the existing binary against the checked-in SASS, creates
the tensor map, loads `candidate.cubin`, launches `bench`, and runs the correctness
check before collecting timings. Output is `run.json` and raw binary samples,
explicitly labeled experimental. The host watchdog cannot guarantee GPU recovery
after a hang; do not use a production training device for initial qualification.

### 4. Plan and run the throughput sweep

After a successful smoke test, set `SM_COUNT` and `L2_BYTES` from that device's
`run.json` fields `device.sm_count` and `device.l2_bytes`; do not use another GPU's
geometry. Generate a plan for a throughput variant:

```sh
python3 -m microbenchmark.blackwell.tma.sweeps \
  --variant tma_tensor_3d_throughput--8x8x8 \
  --sm-count "${SM_COUNT:?Set from smoke-test run.json}" \
  --l2-bytes "${L2_BYTES:?Set from smoke-test run.json}" \
  --global-bytes 536870912 \
  > microbenchmark/blackwell/artifacts/tma-3d-sweep-plan.json
```

This only writes a plan; it does **not** launch kernels. For each JSON entry,
reuse step 3 with its `launch.blocks`, `count`, `repetitions`, `seed`, and
`memory_limit` as `--blocks`, `--count`, `--repetitions`, `--seed`, and
`--memory-limit`. Use `--samples 5 --max-seconds 60` and a fresh `--output` for
each configuration. Run sequentially on the same device, stopping on any failure.

Each throughput variant has four grid sizes: **S, 2S, 3S, 4S** blocks, where S is
the device SM count. Repeat for all 29 throughput variants for **116 configurations**;
the completion-latency variant instead sweeps working-set size. Grid size does not
guarantee a particular per-SM residency. The planner is not a campaign executor.

### 5. Inspect and retain results

- Require `campaign_status: requested_batches_completed`, the requested number
  of `raw_samples`, and each sample's `numerical_check: passed` in `run.json`.
- Inspect `event_ms_statistics`, per-sample `rates`/`cycles`, and telemetry.
  These remain experimental requested-transfer metrics, not peak HBM bandwidth.
- Keep the entire run directory and qualification reports. On failure inspect
  `stderr.txt`, `run.json` (if created), or `timeout.json`; after a timeout, inspect
  the isolated GPU before retrying. Do not bypass correctness or hash checks.

For later imports, follow [the incremental import policy](../README.md). Its
read-only provenance check intentionally reports README edits against the original
import baseline; do not refresh that baseline to hide reviewed changes.
