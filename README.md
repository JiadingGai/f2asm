# F2Asm

F2Asm learns exact NVIDIA SASS instruction encoders using linear algebra over
the binary field F₂. It supports CUBIN disassembly and reassembly across recent
NVIDIA data-center GPU architectures.

> **Status:** Source release in preparation. This repository currently hosts
> the public project page for F2Asm. The implementation and reproducibility
> artifacts will be released after final qualification and licensing review.

## Evaluated targets

- Hopper: `sm_90`, `sm_90a`
- Blackwell: `sm_100`
- Rubin: `sm_107`

## Planned release

- F₂-based SASS instruction encoder
- CUBIN parser, assembler, and rebuilding support
- architecture-specific profiles for Hopper, Blackwell, and Rubin
- training and qualification pipeline
- unit tests and reproducibility manifests

The training corpora and NVIDIA library binaries will not be redistributed.
Reproduction scripts will operate on legally obtained CUDA installations and
library packages.

## Paper

Preprint forthcoming.

## Disclaimer

F2Asm is an independent research project. It is not affiliated with or
endorsed by NVIDIA.
