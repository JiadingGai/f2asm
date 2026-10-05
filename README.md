# F2Asm

F2Asm learns exact NVIDIA SASS instruction encoders using linear algebra over
the binary field F₂. Its goal is to provide a machine-code backend for SASS
coding agents, hand-tuned kernels, and GPU microbenchmarks.

> **Release status: learning-core pilot.** This repository provides the
> self-contained GF(2) learning module and its tests, alongside the previously
> released experimental Rubin encoder artifact. The SASS frontend, repository
> loader, architecture profiles, CLI, and CUBIN tools are not part of this pilot.

## Paper

**[Learning Exact NVIDIA SASS Encoders with F₂ Linear
Algebra](https://arxiv.org/abs/2608.20532)**

## Learning core

[`src/f2asm/bitlinear.py`](src/f2asm/bitlinear.py) uses Python integers as
bitsets for exact Gaussian elimination over F₂. It provides:

- Explicit operand/modifier feature schemas and vector-valued affine models.
- Training with conflicting-observation detection and inference restricted to
  the learned row span; unsupported inputs raise `OutOfSpanError`.
- Incremental extension, same-schema model merging, and JSON serialization.

Exactness is relative to the supplied feature schema and observations. It is
not a claim that arbitrary instruction text or a resulting GPU kernel is valid.
The module uses only the Python standard library; use Python 3.10 or later.

### Try a small example

From a checkout of this repository:

```sh
PYTHONPATH=src python3 - <<'PY'
from f2asm.bitlinear import BitLinearModel, FeatureSchema, OperandField, TrainingSample

# A toy encoding: two operand bits are mapped into a 128-bit output word.
schema = FeatureSchema(operands=(OperandField("x", 2),))
model = BitLinearModel.train(schema, [
    TrainingSample({"x": 0}, 0x10),
    TrainingSample({"x": 1}, 0x11),
])
assert model.encode({"x": 1}) == 0x11
assert not model.supports({"x": 2})

# Add one independent observation without changing the original model.
updated, report = model.extend([TrainingSample({"x": 2}, 0x12)])
assert report.independent_rows_added == 1
assert not model.supports({"x": 2})
restored = BitLinearModel.from_json(updated.to_json())
print(hex(restored.encode({"x": 3})))  # 0x13
PY
```

Run the focused tests with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bitlinear.py' -v
```

This is a source-only pilot, not an installable package or an assembly CLI.
Its single-model JSON retains the `hopperasm.bitlinear` format identifier for
compatibility; that identifier does not imply a dependency on `hopperasm`.

## Artifacts

### Shared Rubin Encoder (Experimental)

We have released an experimental F2Asm encoder model trained jointly for the
NVIDIA Rubin GPU targets `sm_107`, `sm_107f`, and `sm_107a`. It contains 368
instruction forms, 3,428 affine maps, and 54,957 basis rows. All 3,608
controlled held-out CUBINs passed strict round-trip and reproduced every
executable section exactly.

- Artifact: [DefaultBitLinearRepos.sm_107_variants.experimental.json](DefaultBitLinearRepos.sm_107_variants.experimental.json)
- SHA-256: `2478cce78475e173673285092c42d6ed76097038bd6f9ad43cbe171596b1cdec`

**The learning-core module cannot load this repository-format artifact or
assemble SASS text on its own.** Those operations require the unreleased
repository loader and SASS frontend. The artifact's qualification results above
come from the full research implementation, not this pilot alone.

## Research-evaluated targets

The paper's full research implementation evaluated these targets; this list
does not describe target-specific support shipped in the learning-core pilot:

- Hopper: `sm_90`, `sm_90a`
- Blackwell: `sm_100`, `sm_100f`, `sm_100a`, `sm_103`, `sm_103a`
- Rubin: `sm_107`, `sm_107f`, `sm_107a`

## Remaining release scope

- Repository-level training and learned-model loading
- SASS parsing, instruction inference, and architecture-specific profiles
- Fixed-layout CUBIN rebuilding and qualification tools
- Further tests and reproducibility manifests

Training CUBINs and NVIDIA library binaries are not included in this
repository.

## Acknowledgments

F2Asm was inspired by [CuAssembler](https://github.com/cloudcores/CuAssembler).
We thank cloudcores and its contributors.

## Licensing

The released learning-core code and tests are covered by the
[NCSA license](LICENSE).
