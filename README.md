# F2Asm

F2Asm learns exact NVIDIA SASS instruction encoders using linear algebra over
the binary field F₂. It supports CUBIN disassembly and reassembly across recent
NVIDIA data-center GPU architectures. A primary design goal of F2Asm is to
provide a reliable machine-code backend for autonomous SASS coding agents that
generate, optimize, and validate SASS.

## Paper

**[Learning Exact NVIDIA SASS Encoders with F₂ Linear
Algebra](https://arxiv.org/abs/2608.20532)**

> **Status:** Source release in preparation. This repository currently hosts
> the public project page for F2Asm. The implementation and reproducibility
> artifacts will be released after final qualification and licensing review.

## Artifacts

### Shared Rubin Encoder (Experimental)

We have released an experimental F2Asm encoder model trained jointly for the
NVIDIA Rubin GPU targets `sm_107`, `sm_107f`, and `sm_107a`. It contains 368
instruction forms, 3,428 affine maps, and 54,957 basis rows. All 3,608
controlled held-out CUBINs passed strict round-trip and reproduced every
executable section exactly.

- Artifact: `DefaultBitLinearRepos.sm_107_variants.experimental.json`
- SHA-256: `2478cce78475e173673285092c42d6ed76097038bd6f9ad43cbe171596b1cdec`

## Evaluated targets

- Hopper: `sm_90`, `sm_90a`
- Blackwell: `sm_100`, `sm_100f`, `sm_100a`, `sm_103`, `sm_103a`
- Rubin: `sm_107`, `sm_107f`, `sm_107a`

## Planned release

- F₂-based SASS instruction encoder
- CUBIN parser, assembler, and rebuilding support
- architecture-specific profiles for Hopper, Blackwell, and Rubin
- training and qualification pipeline
- unit tests and reproducibility manifests

Training CUBINs and NVIDIA library binaries are not included in this
repository.
