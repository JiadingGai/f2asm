# A 3D TMA transfer, from host to kernel

Example: `tma_tensor_3d_throughput--8x8x8`. Use `count=4` below only to make the
layout easy to follow; actual campaign working sets are larger. `count` is the
number of tiles, **not** the number of thread blocks.

1. **Choose the transfer box.** [catalog.py](catalog.py) defines an 8×8×8 box of
   four-byte elements: each TMA instruction requests 2,048 bytes.
2. **Create and upload the input.** [inputs.py](inputs.py) generates seeded bytes.
   The host's `Driver` initializes CUDA and a B200 context, then allocates and
   uploads `count * 2048 = 8192` bytes using `cuMemAlloc` / `cuMemcpyHtoD`.
   It also allocates `blocks * (32 + 2048)` output bytes.
3. **Describe the global layout.** [Driver.tensor_map](../common/cuda_driver.py)
   treats that input as a tensor with the first dimension contiguous:

   | Descriptor field | Value in this example |
   |---|---|
   | Data type / rank | `UINT32` / 3 |
   | Global address | Uploaded input's device pointer |
   | Global dimensions, in elements | `[32, 8, 8]` |
   | Global strides, in bytes | `[128, 1024]` |
   | Transfer box dimensions | `[8, 8, 8]` |
   | Element strides | `[1, 1, 1]` |
   | Interleave / swizzle / L2 promotion / float OOB mode | All `NONE` |

   Thus `(x,y,z)` addresses `base + 4*x + 128*y + 1024*z`. In C array notation
   this is `uint32_t input[8][8][32]`. One tile reads 64 separate 32-byte rows
   from global memory and packs them into 2 KiB of unswizzled shared memory.
4. **Encode and upload the descriptor.** The host calls `cuTensorMapEncodeTiled`
   with those fields, using a 64-byte-aligned host buffer, then uploads the
   resulting 128-byte opaque tensor map to device memory. Encoding a descriptor
   does not move any tensor data. It is created once per configuration and reused
   across the proof launch and timed samples; it is not rebuilt for each copy.
   See NVIDIA's [CUDA tensor-map API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html).
5. **Load and launch the prepared kernel.** [common/runner.py](../common/runner.py)
   authenticates the offline F2Asm-assembled `candidate.cubin`, loads it with
   `cuModuleLoadData`, and obtains `bench` with `cuModuleGetFunction`.
   `cuLaunchKernel` passes five values: input pointer, output pointer, repetitions,
   tile count, and descriptor device pointer. Grid size is `blocks`; each block
   has 32 threads. The kernel's static shared memory holds the tile and barrier;
   dynamic shared-memory size is zero. No compiler or assembler runs here.
6. **Issue each transfer and wait for completion.** Only thread 0 in each block
   executes this baseline's loop. It initializes a shared-memory barrier and
   performs the async-proxy fence. At iteration `i`, block `b` selects tile
   `(i+b) % count`, with coordinates `(tile*8, 0, 0)`. It registers an expected
   2,048-byte transaction, issues the tensor copy, and polls barrier completion.
   The checked-in [SASS](sass/tma_tensor_3d_throughput--8x8x8.sass) contains
   `UTMALDG.3D` and `SYNCS.PHASECHK.TRANS64.TRYWAIT`. After completion it checksums
   the first shared word, flips the barrier phase, and starts the next iteration.
   A bounded wait traps on timeout. This baseline has one outstanding copy per CTA.
7. **Time and check on the host.** The runner first executes a short correctness
   launch, then event-times each full launch, downloads outputs and validates
   status, checksum, positive cycle counts and every byte of each final tile.
   CPU checking and descriptor construction are outside GPU timing. CUDA events
   include kernel setup and final-tile export; the kernel cycle interval excludes
   that export but includes address/checksum/completion overhead.

The descriptor specifies **layout and box shape**; the instruction supplies
**coordinates, shared-memory destination and completion barrier**. Changing the
tile origin does not require changing the descriptor. Across CTAs, these baseline
streams overlap: requested transfer GB/s must not be labeled distinct HBM bandwidth.
