# Blackwell microbenchmark examples

This directory contains selected experimental examples extracted from the local
`blackwell-microbench` research repository. The first family is [TMA](tma/README.md).
It is a source-checkout example, not an additional dependency of the F2Asm package.
The assembler implementation in `src/f2asm/` is unchanged.

## Incremental import rules

- Each family owns its runner, seed generator, SASS, protocol checks and tests.
  Future memory/tensor families belong in sibling directories, not inside `tma/`.
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
  measurements stay in ignored artifact directories, never copied into Git.
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
