# Lab 02 — Glossary

> Only **new terms** introduced in this lab. Terms defined in
> `labs/lab-01-hello-cuda/GLOSSARY.md` (kernel, kernel launch, stream,
> `cudaMallocAsync`/`cudaFreeAsync`, grid-stride loop, coalesced access,
> `float4`, `__align__(16)`, occupancy, achieved bandwidth, SOL, Memory
> Workload Analysis, RAII, move-only, `std::span`, concept,
> `std::is_trivially_copyable_v`, designated initializers, templated
> kernel) are **not** redefined here.
>
> Read this before [`LAB.md`](./LAB.md) §3 Spec.

## CUDA

- **shared memory** — on-chip, programmer-managed scratchpad memory
  shared by all threads in a block; ~100× lower latency than global
  memory and the prerequisite for tiling.
  First appears in: `src/gemm_tiled_32.cu`.
  See: PMPP 4e §5.4; CUDA C++ Programming Guide §5.3.2.3.
- **tiling** — algorithmic technique of partitioning a large
  computation into small tiles that fit in shared memory or registers,
  amortizing the cost of global-memory loads over many FLOPs.
  First appears in: `src/gemm_tiled_32.cu`.
  See: PMPP 4e §5.4-§5.5.
- **`__syncthreads()`** — block-scope barrier; all threads in the
  block wait until every thread has reached the call before any
  proceeds. Required between shared-memory writes and reads.
  First appears in: `src/gemm_tiled_32.cu`.
  See: PMPP 4e §5.4; CUDA C++ Programming Guide §B.6.
- **bank conflict** — performance penalty when multiple threads in a
  warp address the same shared-memory bank in the same cycle, forcing
  the access to serialize. 32 banks; 4-byte stride; the canonical fix
  is column padding.
  First appears in: `src/gemm_tiled_64.cu` (the padding TODO).
  See: PMPP 4e §5.6; CUDA C++ Best Practices §9.2.3.
- **shared-memory padding** — adding extra columns to a 2D shared tile
  (`__shared__ T tile[TILE][TILE + PAD]`) so the stride between rows
  is no longer a multiple of 32, eliminating bank conflicts.
  First appears in: `src/gemm_tiled_64.cu`.
  See: PMPP 4e §5.6.
- **register tile** — sub-tile of the output computed entirely in a
  thread's registers (e.g. each thread computes a 4×4 block of `C`),
  raising arithmetic intensity by reducing shared-memory traffic.
  First appears in: `src/gemm_tiled_64.cu`.
  See: PMPP 4e §6.4; CUTLASS "Efficient GEMM in CUDA" tutorial.
- **arithmetic intensity** — ratio of FLOPs to bytes loaded
  (FLOP/byte); on the Spark roofline, GEMM crosses from memory-bound
  to compute-bound around ~10 FLOP/byte.
  First appears in: `LAB.md` §5 Hypothesis.
  See: PMPP 4e §5.2; Williams et al., *Roofline* (CACM 2009).
- **`cuda::memcpy_async`** — device-side asynchronous copy primitive
  (CUDA 11+) that issues a global→shared transfer that can overlap
  with compute, using a `cuda::barrier` or `cuda::pipeline` to
  synchronize completion.
  First appears in: `src/gemm_tiled_async.cu`.
  See: CUDA C++ Programming Guide §B.7; libcu++ `<cuda/barrier>`.
- **`cuda::pipeline`** — libcu++ multi-stage barrier abstraction that
  lets a block double-buffer (or N-buffer) `memcpy_async` operations:
  one stage being filled while another is being consumed.
  First appears in: `src/gemm_tiled_async.cu`.
  See: CUDA C++ Programming Guide §B.7; libcu++ `<cuda/pipeline>`.
- **double buffering** — overlapping technique with two shared-memory
  buffers: while threads multiply the data in buffer A, the next tile
  is being copied into buffer B; roles swap each iteration.
  First appears in: `src/gemm_tiled_async.cu`.
  See: PMPP 4e §6.5; CUTLASS GEMM tutorial.
- **GFLOP/s** — billion floating-point operations per second; the
  standard performance metric for compute-bound kernels like GEMM.
  GEMM at `M=N=K` does `2 * M * N * K` FLOPs.
  First appears in: `bench/bench_gemm.cpp`.
  See: PMPP 4e §1.4.
- **cuBLAS** — NVIDIA's closed-source BLAS implementation for CUDA;
  `cublasSgemm` is the single-precision GEMM that defines the perf
  target this week.
  First appears in: `tests/test_gemm.cpp`.
  See: cuBLAS Library User Guide.

## C++20

- **`__restrict__`** — compiler hint (non-standard, supported by
  nvcc/gcc/clang) that two pointer parameters do not alias, enabling
  vectorization and load/store reordering.
  First appears in: `src/gemm_naive.cu`.
  See: CUDA C++ Best Practices Guide §10.3; cppreference
  ["restrict"](https://en.cppreference.com/w/c/language/restrict).
- **strategy / dispatch enum** — Iglberger Ch 5 pattern: an `enum
  class` (here `GemmVersion`) selects which implementation runs at
  the call site, decoupling client code from the kernel-version
  decision tree.
  First appears in: `src/gemm.hpp` and `src/gemm_launch.cu`.
  See: Iglberger Ch 5 ("Strategy modernization").

## Spatial Intelligence / CV

*No new terms introduced this week.*
*Month 1 is pure C++20 + CUDA foundations; CV / Spatial Intelligence
vocabulary starts in Month 3.*

## Python bindings

- **`torch.utils.cpp_extension`** — PyTorch's first-party mechanism
  for compiling and loading C++/CUDA extensions that expose `torch::
  Tensor`-typed ops to Python. Two modes: JIT (`load()`) and AOT
  (`setup.py` + `BuildExtension`). Month 1-2 uses JIT.
  First appears in: `python/gemm_ext.py`.
  See: [PyTorch C++ extension docs](https://pytorch.org/tutorials/advanced/cpp_extension.html);
  `.cursor/skills/python-bindings/SKILL.md`.
- **JIT loader (`cpp_extension.load`)** — compiles the listed `.cpp`
  / `.cu` sources on first import and caches the resulting `.so` by
  source hash; lets a lab evolve without a `setup.py`. Used in
  Months 1-2 only.
  First appears in: `python/gemm_ext.py`.
  See: PyTorch docs;
  `.cursor/skills/python-bindings/SKILL.md` Pattern A.
- **`PYBIND11_MODULE`** — pybind11 macro that registers a function as
  a Python-callable symbol on the extension module. Used in Months
  1-2; replaced by `TORCH_LIBRARY` in Month 3+ for `torch.compile` /
  TensorRT compatibility.
  First appears in: `src/gemm_pybind.cpp`.
  See: pybind11 docs;
  `.cursor/skills/python-bindings/SKILL.md`.
- **`<torch/extension.h>`** — umbrella header that pulls in the
  `torch::Tensor` API, the `TORCH_CHECK` macro, and the pybind11
  glue used by JIT extensions.
  First appears in: `src/gemm_pybind.cpp`.
  See: PyTorch C++ API docs.
- **`at::cuda::getCurrentCUDAStream()`** — accessor that returns the
  CUDA stream PyTorch is currently using on this thread; custom ops
  must thread it into their kernel launches so `torch.cuda.Stream`
  contexts work as expected.
  First appears in: `src/gemm_pybind.cpp`.
  See: PyTorch C++ API; `.cursor/skills/python-bindings/SKILL.md`
  ("Wrong stream" pitfall).
- **`TORCH_CHECK`** — runtime assertion macro that throws a Python
  `RuntimeError` with the supplied message if the condition fails;
  the canonical input-validation tool inside a custom op.
  First appears in: `src/gemm_pybind.cpp`.
  See: PyTorch C++ API.
- **wrapper overhead** — additional Python-side latency per op call
  (dispatch, type checks, allocator hits, stream lookup) on top of
  raw kernel time; the lab targets < 5% at the largest test size.
  First appears in: `python/test_gemm.py` (`test_overhead_bound`).
  See: `.cursor/skills/python-bindings/SKILL.md`.
