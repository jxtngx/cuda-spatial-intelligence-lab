# Lab 03 — Glossary

> Only **new terms** introduced in this lab. Terms defined in
> `labs/lab-01-hello-cuda/GLOSSARY.md` (kernel, kernel launch,
> stream, `cudaMallocAsync`/`cudaFreeAsync`, grid-stride loop,
> coalesced access, `float4`, `__align__(16)`, occupancy, achieved
> bandwidth, SOL, Memory Workload Analysis, RAII, move-only,
> `std::span`, concept, `std::is_trivially_copyable_v`, designated
> initializers, templated kernel) and in
> `labs/lab-02-tiled-gemm/GLOSSARY.md` (shared memory, tiling,
> `__syncthreads`, bank conflict, shared-memory padding, register
> tile, arithmetic intensity, `cuda::memcpy_async`, `cuda::pipeline`,
> double buffering, GFLOP/s, cuBLAS, `__restrict__`, strategy/dispatch
> enum, `torch.utils.cpp_extension`, JIT loader, `PYBIND11_MODULE`,
> `<torch/extension.h>`, `at::cuda::getCurrentCUDAStream`,
> `TORCH_CHECK`, wrapper overhead) are **not** redefined here.
>
> Read this before [`LAB.md`](./LAB.md) §3 Spec.

## CUDA

- **reduction** — collective op that folds an array of `n` values
  under an associative binary op (typically `+`) into a single value
  in `O(log n)` parallel span.
  First appears in: `src/reduce.hpp`.
  See: PMPP 4e §10; Mark Harris, *Optimizing Parallel Reduction in CUDA*.
- **inclusive scan / prefix sum** — `out[i] = in[0] + in[1] + ... + in[i]`
  for all `i`. The exclusive variant excludes `in[i]`. The fundamental
  primitive behind RLE, sort, segmentation boundaries.
  First appears in: `src/scan.hpp`.
  See: PMPP 4e §11; Blelloch, *Prefix Sums and Their Applications*.
- **Hillis-Steele scan** — single-pass scan that, at step `s`, has each
  thread add `in[tid - 2^s]` to `in[tid]`. Work O(n log n), span
  O(log n); inferior in work to Brent-Kung but trivial to implement.
  First appears in: `src/scan.cu` (`scan_hillis_steele`).
  See: PMPP 4e §11.2; Hillis & Steele 1986.
- **Brent-Kung scan** — work-efficient O(n) scan that sweeps an upsweep
  (reduction) tree then a downsweep (distribution) tree. Stretch goal
  for the week; baseline in CUB.
  First appears in: `LAB.md` §10.
  See: PMPP 4e §11.3; Brent & Kung 1979.
- **warp shuffle (`__shfl_down_sync` / `__shfl_xor_sync`)** — register-
  to-register data exchange among the 32 threads of a warp via the
  shuffle instruction; no shared memory and no `__syncthreads()`
  required. Foundation of the v4 reduction.
  First appears in: `src/reduce.cu` (`warp_reduce_sum`).
  See: CUDA C++ Programming Guide §B.16; Harris slide 35.
- **lane** — the index of a thread within its warp, `tid & 31`.
  First appears in: `src/reduce.cu` (`reduce_v4`).
  See: CUDA C++ Programming Guide §5.4.
- **cooperative groups** — CUDA library (`<cooperative_groups.h>`) that
  exposes named, statically-typed thread groupings (`thread_block`,
  `tiled_partition<32>`, `coalesced_threads()`) and collectives
  (`reduce`, `inclusive_scan`) over them.
  First appears in: `src/scan.cu` (`scan_coop_groups`).
  See: CUDA C++ Programming Guide §B.18; CCCL `cooperative_groups`.
- **`cooperative_groups::inclusive_scan`** — collective inclusive scan
  over an arbitrary cooperative group; on a `tiled_partition<32>` it
  is implemented entirely with warp shuffles.
  First appears in: `src/scan.cu`.
  See: CUDA C++ Programming Guide §B.18.5;
  `<cooperative_groups/scan.h>`.
- **atomic operation (`atomicAdd`)** — read-modify-write to global or
  shared memory that is guaranteed serializable across threads. Cheap
  on shared memory; expensive on global memory under contention.
  First appears in: `src/reduce.cu` (`reduce_v4` final accumulator),
  `src/histogram.cu`.
  See: PMPP 4e §9; CUDA C++ Programming Guide §B.14.
- **histogram (parallel)** — count occurrences of each bin value in
  the input; canonical pattern for atomics + privatization.
  First appears in: `src/histogram.hpp`.
  See: PMPP 4e §9.
- **privatization** — give each block (or warp) its own copy of the
  reduction target in shared memory, accumulate locally, then merge
  to the global target once at the end. Turns N global atomics into
  ~`gridDim.x * BINS` global atomics.
  First appears in: `src/histogram.cu` (`hist_shared_warp`).
  See: PMPP 4e §9.4.
- **warp aggregation** — within a warp, group threads writing to the
  same target (via `__match_any_sync` or `__ballot_sync`), elect one
  lane per group, and have that lane do a single `atomicAdd` of the
  group size. Turns `WARP` atomics into `≤ WARP` (often just 1).
  First appears in: `src/histogram.cu` (the marked TODO).
  See: PMPP 4e §9.5; NVIDIA blog
  *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics*.
- **`__ballot_sync`** — warp vote primitive that returns a 32-bit
  mask whose bit `i` is set iff lane `i`'s predicate is true. Used to
  identify lanes participating in a warp-aggregated atomic.
  First appears in: `src/histogram.cu` (TODO).
  See: CUDA C++ Programming Guide §B.15.
- **`__match_any_sync`** — warp vote primitive that returns, for each
  lane, the mask of other lanes in the warp whose value equals this
  lane's value. The ideal primitive for warp-aggregated histograms.
  First appears in: `src/histogram.cu` (TODO).
  See: CUDA C++ Programming Guide §B.15.
- **CUB** — CCCL's CUDA collective primitives library
  (`cub::DeviceReduce`, `cub::DeviceScan`, `cub::WarpReduce`,
  `cub::BlockReduce`). Defines the perf baselines this week.
  First appears in: `src/reduce_cub.cu`, `src/scan_cub.cu`.
  See: [CUB docs](https://nvidia.github.io/cccl/cub/);
  CCCL repo.
- **`cub::DeviceReduce::Sum`** — device-wide sum reduction; the perf
  target for `reduce_v4` (≥ 80% throughput).
  First appears in: `src/reduce_cub.cu`.
  See: CUB docs.
- **`cub::DeviceScan::InclusiveSum`** — device-wide inclusive scan;
  the perf target for the best Week-03 scan (≥ 70% throughput).
  First appears in: `src/scan_cub.cu`.
  See: CUB docs.
- **single-pass / two-pass reduction** — single-pass: one kernel
  launch reduces `n` → 1 via grid-stride + `atomicAdd` of warp totals
  (this is `reduce_v4`). Two-pass: kernel A reduces blocks → partials,
  kernel B reduces partials → 1 (this is the v0..v3 host launcher).
  First appears in: `src/reduce.cu` (`launch_reduce`).
  See: PMPP 4e §10.5; CUB `DeviceReduce` source.

## C++20

*No new terms introduced this week.*

The `enum class` + dispatch pattern, `__restrict__`, lambdas, and
`std::span` all date to Weeks 01-02 and are reused here.

## Spatial Intelligence / CV

*No new terms introduced this week.*

These primitives (reduction, scan, histogram) are CV-adjacent —
prefix sums underpin integral images, histograms underpin classical
histogram-equalization and feature descriptors — but the concrete CV
applications are introduced from Month 3 onward.

## Python bindings

*No new terms introduced in this lab.*

Same JIT loader / `PYBIND11_MODULE` / `at::cuda::getCurrentCUDAStream`
pattern as Lab 02. The only new wrinkle (one extension exposing
multiple primitive families behind one `_ext` module) is a layout
choice, not new vocabulary.
