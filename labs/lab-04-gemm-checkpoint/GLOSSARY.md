# Lab 04 — Glossary

> Only **new terms** introduced in this lab. Terms defined in
> `labs/lab-01-hello-cuda/GLOSSARY.md`,
> `labs/lab-02-tiled-gemm/GLOSSARY.md`, and
> `labs/lab-03-reduce-scan/GLOSSARY.md` are **not** redefined here.
> In particular: shared memory, tiling, `__syncthreads`, bank
> conflict, shared-memory padding, register tile, arithmetic
> intensity, `cuda::memcpy_async`, `cuda::pipeline`, double
> buffering, GFLOP/s, cuBLAS, `__restrict__`, strategy/dispatch
> enum, `float4`, occupancy, achieved bandwidth, SOL, Memory
> Workload Analysis, `torch.utils.cpp_extension`, JIT loader,
> `PYBIND11_MODULE`, `<torch/extension.h>`,
> `at::cuda::getCurrentCUDAStream`, `TORCH_CHECK`, and wrapper
> overhead all carry over unchanged.
>
> Read this before [`LAB.md`](./LAB.md) §3 Spec.

## CUDA

- **`__ldg(ptr)`** — intrinsic that issues a global load through
  the read-only data cache (the same cache backing `const __restrict__`
  loads on modern architectures). Cheap optimization for inputs
  that are read-many / written-never within a kernel, like the `A`
  and `B` operands of GEMM.
  First appears in: `src/gemm_v4_checkpoint.cu` (TODO 2).
  See: CUDA C++ Programming Guide §10.7;
  CUDA C++ Best Practices Guide §9.2.4.
- **roofline model** — performance-bound diagram with arithmetic
  intensity (FLOP / byte) on the x-axis and attainable GFLOP/s on
  the y-axis; the "roof" is the min of the DRAM bandwidth line
  (slope = peak GB/s) and the compute peak (horizontal = peak
  GFLOP/s). A kernel sitting on the diagonal is memory-bound; one
  sitting on the ceiling is compute-bound. The Week-04 writeup
  must place every variant on this plot.
  First appears in: `LAB.md` §5 Hypothesis, `report/roofline.png`.
  See: Williams, Waterman, Patterson — *Roofline: An Insightful
  Visual Performance Model*, CACM 2009; PMPP 4e §5.2.
- **thread-block tile / warp tile / MMA tile (CUTLASS hierarchy)**
  — three-level decomposition of a GEMM: the thread-block tile is
  the BM×BN×BK chunk of `C` a CTA computes; the warp tile is the
  WM×WN×WK chunk a single warp computes from registers; the MMA
  tile is the smallest matrix-multiply-accumulate the hardware
  exposes (e.g. 16×8×16 on tensor cores; 4×4×1 in our SIMT
  register-tile this week). Vocabulary borrowed from CUTLASS.
  First appears in: `LAB.md` §1 (CUDA), `src/gemm_v4_checkpoint.cu`
  comments.
  See: CUTLASS *Efficient GEMM in CUDA* tutorial.
- **CUTLASS** — NVIDIA's open-source CUDA C++ template library of
  GEMM, conv, and attention primitives, organized around the
  thread-block / warp / MMA tile hierarchy. Used here only as a
  *sanity baseline* via `cutlass::gemm::device::Gemm<...>`; the
  primary CUTLASS lab is Lab 05.
  First appears in: `src/cutlass_baseline.cu`.
  See: https://github.com/NVIDIA/cutlass; CUTLASS docs.
- **`cutlass::gemm::device::Gemm<...>`** — the device-level GEMM
  template entry point in CUTLASS; instantiated with element
  types, layouts, op class (SIMT vs TensorOp), and tile shapes.
  First appears in: `src/cutlass_baseline.cu` (TODO).
  See: CUTLASS docs, `examples/00_basic_gemm`.
- **read-only data cache** — the on-SM L1-resident cache that
  serves loads marked read-only (`const __restrict__` /
  `__ldg`); independent of the L1 data cache for stores. The
  optimization target of `v4c`.
  First appears in: `src/gemm_v4_checkpoint.cu`.
  See: CUDA C++ Programming Guide §5.3.2.4;
  CUDA C++ Best Practices Guide §9.2.4.
- **vectorized load** — single instruction that fetches 4, 8, or
  16 bytes per thread (`float4`, `float2`, `int4`) into registers
  or shared memory, halving (or quartering) the issued load count
  vs the equivalent scalar loop. Required for SGEMM to saturate
  DRAM throughput on Blackwell.
  First appears in: `src/gemm_v4_checkpoint.cu` (TODO 1).
  See: CUDA C++ Programming Guide §6.2.4;
  CUDA C++ Best Practices Guide §9.2.1.
- **launch geometry sweep** — practice of compiling the same
  kernel under multiple `(BM, BN, BK)` tile-shape parameters as
  separate template instantiations, benchmarking all of them, and
  picking the winner empirically rather than analytically. Made
  ergonomic by `if constexpr` + a non-type `TileConfig` template
  parameter.
  First appears in: `src/gemm_launch.cu` (`launch_v4_checkpoint`).
  See: PMPP 4e §6.4; CUTLASS GEMM tutorial.
- **register pressure / register spill** — when a thread's live
  values exceed the 255-register-per-thread limit on Blackwell,
  the compiler "spills" excess live values to local memory
  (DRAM-backed); this nukes performance. Reported by
  `nvcc -Xptxas=-v` and visible in Nsight Compute under
  *Launch Statistics → Registers Per Thread*. The 4×4 register
  tile this week is precisely the optimization at risk of spills.
  First appears in: `LAB.md` §1 (CUDA outcome 3),
  `src/gemm_v4_checkpoint.cu`.
  See: CUDA C++ Best Practices Guide §10.2; Nsight Compute User
  Guide.

## C++20

- **non-type template parameter (`TileConfig` struct as NTTP)** —
  C++20 lets a literal-type `struct` be passed as a template
  argument; the lab uses
  `template <TileConfig C> __global__ void gemm_v4_kernel(...)`
  so each tile size compiles to its own kernel without
  preprocessor macros.
  First appears in: `src/gemm_v4_checkpoint.cu`.
  See: cppreference
  ["Non-type template parameter"](https://en.cppreference.com/w/cpp/language/template_parameters);
  P0732R2 "Class types in non-type template parameters".
- **`if constexpr` dispatch on tile shape** — use of `if constexpr`
  inside a kernel template body to specialize the inner loop on
  compile-time tile parameters without runtime branches and
  without macro hell.
  First appears in: `src/gemm_v4_checkpoint.cu`,
  `src/gemm_launch.cu`.
  See: cppreference
  ["if constexpr"](https://en.cppreference.com/w/cpp/language/if).

## Spatial Intelligence / CV

*No new terms introduced this week.*

GEMM is the substrate every Month-3 lab leans on (NeRF MLPs,
Gaussian-splat projection, attention QKV), but the spatial-
intelligence vocabulary itself begins in Lab 09.

## Python bindings

*No new terms introduced this week.*

The JIT-loader pattern, `PYBIND11_MODULE` glue, stream threading,
and the `< 5%` wrapper-overhead test are all carry-overs from
Weeks 02-03. The only new wrinkle this week — registering a fifth
`GemmVersion` enum value in `_VERSION_MAP` — is a layout choice,
not new vocabulary.
