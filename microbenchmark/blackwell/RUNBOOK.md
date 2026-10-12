# Runpod B200 runbook

Run local commands from the checkout root; configure `ssh runpod` privately.
Requires an isolated B200, Python 3.10+, NVIDIA driver, 8 GiB available host RAM
and 30 GB disk. Prepare the 31 CUBINs using the [TMA](tma/README.md#2-build-and-assemble-offline)
and [ordinary-load](memory/README.md#2-compile-and-assemble-offline) build guides.
The public learning-core pilot alone cannot assemble them.

Default campaign: **216 configurations, five samples each**, one GPU worker.
No compiler, assembler or extra Python packages are needed on the GPU host.

## 1. Stage from the Mac

Use a fresh remote directory; never overwrite an earlier run. These commands use
the existing qualified bundles. Match local/remote SHA-256 values before proceeding.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s microbenchmark/blackwell -t . -q
COPYFILE_DISABLE=1 tar --no-xattrs --exclude='*/__pycache__' \
  --exclude='microbenchmark/blackwell/artifacts' -czf /tmp/f2asm-tma-source.tar.gz \
  microbenchmark/__init__.py microbenchmark/blackwell
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/f2asm-tma-qualified.tar.gz \
  microbenchmark/blackwell/artifacts/tma-checker-cfg-20261011 \
  microbenchmark/blackwell/artifacts/memory-checker-cfg-20261011
shasum -a 256 /tmp/f2asm-tma-source.tar.gz /tmp/f2asm-tma-qualified.tar.gz
ssh runpod 'test ! -e /workspace/f2asm-tma-run && mkdir /workspace/f2asm-tma-run'
scp /tmp/f2asm-tma-source.tar.gz /tmp/f2asm-tma-qualified.tar.gz runpod:/workspace/f2asm-tma-run/
ssh runpod 'cd /workspace/f2asm-tma-run && sha256sum *.tar.gz'
```

## 2. Check and run on the pod

Confirm the GPU is idle and container RAM/disk are sufficient before launching.

```sh
ssh runpod
cd /workspace/f2asm-tma-run
nvidia-smi
df -h /workspace
free -h
cat /sys/fs/cgroup/memory.max
tar -xzf f2asm-tma-source.tar.gz
tar -xzf f2asm-tma-qualified.tar.gz
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LC_ALL=C
python3 -m unittest discover -s microbenchmark/blackwell -t . -q
python3 -m microbenchmark.blackwell.campaign --probe-device 0
nohup python3 -u -m microbenchmark.blackwell.campaign --suite tma \
  --qualified tma=microbenchmark/blackwell/artifacts/tma-checker-cfg-20261011 \
  --qualified memory=microbenchmark/blackwell/artifacts/memory-checker-cfg-20261011 \
  --output results/tma-b200-campaign --device 0 --samples 5 \
  --max-seconds 60 --worker-timeout 300 \
  --execute --accept-unvalidated-hardware-protocols \
  > campaign-console.log 2>&1 < /dev/null &
tail -n 12 campaign-console.log
```

Failures stop the campaign. Inspect the failed attempt's logs and GPU health.
Resume the identical command with `--resume`; failed/interrupted work also needs
`--retry-failed` after inspection/recovery. Changed code/settings require a fresh
directory. Never bypass checks or automatically reset the GPU.

## 3. Verify and download before stopping the pod

Require `state.json` status `completed`, **216 passed jobs and 1,080 passed samples**.
On the pod, preserve the entire staged workspace:

```sh
python3 -m microbenchmark.blackwell.campaign --verify-only --output results/tma-b200-campaign
cd /workspace
tar -czf f2asm-tma-run-results.tar.gz f2asm-tma-run
sha256sum f2asm-tma-run-results.tar.gz
exit
```

Back on the Mac, use a fresh destination; match the archive checksum, then validate:

```sh
D=microbenchmark/blackwell/artifacts/runpod-b200-results
mkdir "$D"
scp runpod:/workspace/f2asm-tma-run-results.tar.gz "$D/"
shasum -a 256 "$D/f2asm-tma-run-results.tar.gz"
tar -xzf "$D/f2asm-tma-run-results.tar.gz" -C "$D"
R="$D/f2asm-tma-run/results/tma-b200-campaign"
python3 -m microbenchmark.blackwell.campaign --verify-only --output "$R"
```

**Stop the pod only after successful local verification.** Keep the full archive,
not just CSVs/plots. Results contain summaries, samples and raw evidence under `runs/`.
`--report-only` regenerates reports, including incomplete ones; it is not verification.

## 4. Plot offline (optional)

Use Python 3.12+ for the plotting environment; `R` is the downloaded results above.

```sh
python3.12 -m venv /tmp/f2asm-plot-env
/tmp/f2asm-plot-env/bin/python -m pip install matplotlib==3.11.2
MPLCONFIGDIR=/tmp/f2asm-mplconfig /tmp/f2asm-plot-env/bin/python \
  -m microbenchmark.blackwell.plots.paper_figures --results "$R" --output output/pdf/b200-tma
```

Exports Figures 2-4 as PNG/SVG/PDF, plotted values and provenance hashes.

## Recorded B200 run: 2026-10-11

216/216 configurations and 1,080 samples passed in **13m 40s**; completed resume
launched no workers. B200: 148 SMs, 126.5 MiB L2; driver 570.211.01, Python 3.12.3.
An initial Python large-buffer overflow was fixed with chunked input generation;
both attempts were preserved. All 81 CPU tests passed; SASS and CUBINs were unchanged.

Local results:
`artifacts/runpod-b200-20261011/f2asm-tma-20261011-v2/results/tma-b200-campaign/`
(relative to this runbook). The [downloaded archive](artifacts/runpod-b200-20261011/f2asm-tma-20261011-v2-results.tar.gz)
was verified locally against SHA-256
`894a027c9ff20e73205f98ed1ef0f5569d7ccdbf8ef404e988c875fe40d16ef9`.

## Measurement limits

Throughput uses overlapping streams: report **requested transfer GB/s, not HBM
bandwidth**. LDG/TMA node spacing and warm-up differ; subtraction is not pure TMA
overhead. Timings include loop/completion overhead and, for CUDA events, final-tile
export. Gather4, CTA-pair and new swizzle tests are not included.
