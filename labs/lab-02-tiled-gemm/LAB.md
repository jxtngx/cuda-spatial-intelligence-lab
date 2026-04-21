# Lab 02 — Memory hierarchy and tiled GEMM

> Tier **A** scaffold. See [`labs/_template/README.md`](../_template/README.md)
> for what that means: code compiles as-is, with 2-4 `// TODO(student):`
> markers per file. Tests + bench + CMake fully provided.

## 0. Intro

This is the most important week of Month 1. Last week your AXPY kernel
saturated DRAM bandwidth — that was easy because every byte was used
once. This week you implement a single-precision GEMM
(`C = alpha * A @ B + beta * C`, M=N=K=4096) four different ways and
watch the same problem go from compute-starved to bandwidth-bound to
nearly cuBLAS-class as you progressively refactor it. The lesson is
the GPU memory hierarchy: global → shared → register, and the tools
(shared-memory tiling, async copies, padding for bank conflicts) that
move data between them. By Friday you will have a `gemm_tiled_async`
kernel hitting **≥ 50% of `cublasSgemm`** at 4096³, a Nsight Compute
report for each version explaining *why* it moved the needle, and —
**for the first time in the curriculum** — a PyTorch custom op
wrapping that kernel, with a `pytest` proving the numerics from
Python and that the wrapper overhead is < 5% of kernel time.

> **New terms this week.** See [`GLOSSARY.md`](./GLOSSARY.md) in this
> folder. Read it before §3 Spec — it covers the CUDA, C++20, CV /
> Spatial Intelligence, and Python-bindings terms that are introduced
> for the first time this week.

## Plan of work — order of operations

Work this lab top-to-bottom. Don't move past a checkbox until it's
green. Phase letters are referenced by `/checkpoint` when grading.

### A. Read first (do not skip)

- [ ] PMPP 4e Ch 4 (Compute architecture and scheduling).
- [ ] PMPP 4e Ch 5 (Memory architecture and data locality) **end-to-end**.
- [ ] CUDA C++ Best Practices Guide §9 (Memory optimizations) end-to-end.
- [ ] Iglberger Ch 4-5 (Visitor / Strategy modernization).
- [ ] CUDA C++ Programming Guide §B.7 (`cuda::memcpy_async`,
      `cuda::pipeline`).
- [ ] Skim this lab's [`GLOSSARY.md`](./GLOSSARY.md).

### B. Bring the scaffold up on Spark

- [ ] Toolchain check (CUDA 13, CMake ≥ 3.28, Ninja, cuBLAS,
      Nsight host tools, sm_121).
- [ ] From this folder: `cmake -S . -B build -G Ninja && cmake --build build -j`.
- [ ] Baseline `ctest --test-dir build --output-on-failure`. The
      reference variants compile but the `// TODO(student):` blocks
      mean some cases will fail until Phase D — that is expected.

### C. Write §5 Hypothesis *before* you optimize

- [ ] Predict the dominant bottleneck for v0 / v1 / v2 / v3 from
      readings (memory-bound vs compute-bound, expected GFLOP/s as
      a fraction of cuBLAS, which Nsight Compute counter will
      confirm). Commit it to §5 in writing **before** you measure.

### D. Implement the §4 TODOs and turn ctest green

- [ ] `src/gemm_naive.cu` — fill the inner `K` multiply-accumulate.
- [ ] `src/gemm_tiled_32.cu` — declare 32×32 shared tiles,
      load/`__syncthreads`/multiply/`__syncthreads`, write the
      `alpha * acc + beta * C[row,col]` epilogue.
- [ ] `src/gemm_tiled_64.cu` — pick the padding column count that
      kills 32-way bank conflicts, fill the `acc[4][4]` register tile.
- [ ] `src/gemm_tiled_async.cu` — issue `cuda::memcpy_async` for the
      *next* tile before the current-tile multiply, and place
      `pipeline.consumer_wait/release()` correctly.
- [ ] `ctest --test-dir build --output-on-failure` — every
      parametrized size (128, 512, 1024, 4096) must pass for all
      four versions.

### E. Get the Python bindings green

- [ ] `pytest python/` — numerics vs `torch.matmul` at sizes
      {128, 512, 1024, 4096} and the wrapper-overhead bound.
- [ ] If overhead exceeds 5%, follow
      `.cursor/skills/python-bindings/SKILL.md` to diagnose
      (synchronization, host-side allocation, autograd) before
      blaming the kernel.

### F. Run the bench and hit the perf target

- [ ] `./build/bench_gemm` on Spark, capture ms / GFLOP/s / % cuBLAS
      for each version into the §7 Results table.
- [ ] **Target: `gemm_tiled_async` ≥ 50% of `cublasSgemm` at
      M=N=K=4096.** If you miss it, iterate (block shape, async
      pipeline depth, register pressure) and document each attempt
      in §8.
- [ ] **Target: Python wrapper overhead < 5%** of kernel time at
      M=N=K=4096. Capture the numbers in §7.

### G. Profile (evidence for the perf claim)

Follow `.cursor/skills/nsight-profiling/SKILL.md`. Commit raw
artifacts under `report/`.

- [ ] `report/ncu_gemm_v0.ncu-rep` — Memory Workload Analysis +
      Speed of Light, full sections.
- [ ] `report/ncu_gemm_v1.ncu-rep` — same + shared-bank-conflict
      counters.
- [ ] `report/ncu_gemm_v2.ncu-rep` — same + occupancy section
      (register-tile cost).
- [ ] `report/ncu_gemm_v3.ncu-rep` — same + the `cuda::memcpy_async`
      overlap evidence.
- [ ] (Optional but recommended) `report/nsys_gemm.qdrep` —
      timeline of one bench run.

### H. Write it up

- [ ] Fill §7 Results table from the bench output (do not omit any
      version).
- [ ] §8 Discussion: cite the Nsight section + counter for each
      claim in §5 Hypothesis. Honesty about misses counts.
- [ ] §10 What I would do next.
- [ ] Run `/lab-report` to polish (`.cursor/skills/lab-notebook/SKILL.md`).

### I. Self-grade

- [ ] `/checkpoint` against the 5-axis rubric in
      `.cursor/skills/weekly-checkpoint/SKILL.md`. **≥ 14/20** to
      advance.

### Definition of done for Lab 02

`ctest` is green at all four sizes for all four versions; `pytest
python/` is green; `bench_gemm` shows `gemm_tiled_async` ≥ 50% of
`cublasSgemm` at 4096³ and Python wrapper overhead < 5%; `report/`
holds four `ncu_gemm_v{0,1,2,3}.ncu-rep` files; `LAB.md` §5, §7,
§8, §10 are written.

## 1. What you will learn

By the end of this lab you will be able to:

### CUDA
- Articulate the GPU memory hierarchy (global / L2 / shared / register)
  in bandwidth and latency numbers, not as vibes.
- Write a shared-memory tiled GEMM and explain *why* the tile size
  trades off arithmetic intensity vs occupancy.
- Diagnose and eliminate shared-memory **bank conflicts** with padding.
- Use `cuda::memcpy_async` (and `cuda::pipeline`) to overlap global→shared
  copies with compute.
- Read a Nsight Compute report: "Memory Workload Analysis", "Compute
  Workload Analysis", and "Source Counters" sections.

### C++20
- Use `std::mdspan`-style 2D indexing (or a hand-rolled equivalent) to
  keep matrix code readable without giving up `__restrict__`.
- Apply concept-constrained kernel templates so `gemm<T>` works for
  `float` now and `__half` later.
- Use designated-initializer config structs (`GemmConfig{.M=..., .N=...}`)
  the same way last week's `LaunchConfig` did.

### CV / Spatial Intelligence
*N/A this week — Month 1 is pure C++20 + CUDA foundations; CV/SI
vocabulary starts in Month 3.*

### Python bindings
- Wrap `gemm_tiled_async` as a PyTorch custom op via
  `torch.utils.cpp_extension.load()` (JIT loader — Pattern A in
  `.cursor/skills/python-bindings/SKILL.md`).
- Verify numerics from Python against `torch.matmul` on CPU.
- Measure that wrapper overhead is < 5% of kernel time at M=N=K=4096.
- Understand why the wrapper must thread
  `at::cuda::getCurrentCUDAStream()` through the launcher.

## 2. Prerequisites

**Readings (do these first):**
- PMPP 4e Ch 4 (Compute architecture and scheduling), **Ch 5
  (Memory architecture and data locality) end-to-end**.
- CUDA C++ Best Practices Guide §9 (Memory optimizations) end-to-end.
- Iglberger Ch 4-5 (Visitor / Strategy modernization).
- CUDA C++ Programming Guide §B.7 (asynchronous copy) and the
  `cuda::pipeline` section.

**Prior labs' deliverables you need working:**
- Lab 01: `Stream`, `DeviceBuffer<T>`, `CUDA_CHECK`, the
  `launch<>` helper, and your AXPY profiling workflow. The Lab 02
  CMake imports them from `../lab-01-hello-cuda/src` indirectly via
  copies of the headers — each lab is self-contained.

**Toolchain checks:**
- `nvidia-smi`, `nvcc --version`, CMake ≥ 3.28.
- cuBLAS available (it ships with the CUDA toolkit; CMake links via
  `CUDA::cublas`).
- A working Python env with `torch` (CUDA-enabled) and `pytest` for
  the `python/` portion. See
  [`docs/GETTING-STARTED.md`](../../docs/GETTING-STARTED.md).

## 3. Spec

### Inputs / outputs / dtypes
- Compute `C = alpha * A @ B + beta * C` where `A` is `M×K`, `B` is
  `K×N`, `C` is `M×N`, all **row-major**, `float` (single precision).
- Default size: `M = N = K = 4096`. Tests cover smaller sizes too
  (`128`, `512`, `1024`).
- `alpha`, `beta` are `float` scalars. The `pytest` uses
  `alpha = 1.0, beta = 0.0` for simplicity; the C++ tests cover
  `alpha != 1, beta != 0`.

### Numerical tolerance
- `max_abs_error <= 1e-2` against a CPU reference (`std::vector` triple
  loop) at sizes ≤ 1024.
- `max_abs_error / max_abs(C_ref) <= 1e-3` (relative) at 4096³ against
  `cublasSgemm`. (Single precision accumulating 4096 multiplies has
  measurable rounding; this is the right tolerance.)

### Performance target
- **`gemm_tiled_async` ≥ 50% of `cublasSgemm`** at M=N=K=4096 on Spark
  (sm_121). Lab 04's checkpoint will push this to ≥ 70%.
- Python wrapper overhead **< 5%** of kernel time at M=N=K=4096.
- All four versions must produce a Nsight Compute report committed
  under `report/`.

## 4. Your task

This week is **Tier A**: every file in `src/` and `python/` already
compiles. You finish 2-4 marked `// TODO(student):` blocks per file.
Tests, bench, and CMake are provided in full.

- `src/gemm.hpp` — provided. Declares `enum class GemmVersion { v0_naive,
  v1_tiled32, v2_tiled64_padded, v3_tiled_async }` and the
  `gemm<float>(...)` host launcher signature. **Do not modify.**
- `src/gemm_naive.cu` — one TODO: fill the inner-product loop body
  (the multiply-accumulate over `K`).
- `src/gemm_tiled_32.cu` — three TODOs: (a) declare the 32×32 shared
  tiles for `A` and `B`, (b) write the load-tile / `__syncthreads` /
  multiply-tile / `__syncthreads` body, (c) write the
  `C[row, col] = alpha * acc + beta * C[row, col]` epilogue.
- `src/gemm_tiled_64.cu` — two TODOs: (a) choose the **padding column
  count** that eliminates 32-way bank conflicts on the 64-wide tile,
  (b) fill the `acc[4][4]` register-tile inner loop. (You compute a
  4×4 sub-tile per thread; loops + types are provided.)
- `src/gemm_tiled_async.cu` — two TODOs: (a) issue the
  `cuda::memcpy_async` for the *next* tile inside the loop *before*
  the multiply on the *current* tile, (b) call
  `pipeline.consumer_wait()` / `pipeline.consumer_release()` in the
  right places. The pipeline + barrier scaffolding is provided.
- `src/gemm_pybind.cpp` — provided. Implements `gemm_py` exactly per
  `.cursor/skills/python-bindings/SKILL.md` Pattern A. Threads
  `at::cuda::getCurrentCUDAStream()` through. **Do not modify** —
  this is the canonical pybind shape you will copy from in Weeks 3-16.
- `src/gemm_launch.cu` — provided. Dispatches `GemmVersion` to the
  right kernel. **Do not modify.**
- `tests/test_gemm.cpp` — provided. GoogleTest cases for each version
  vs CPU reference (sizes 128, 512, 1024) and vs `cublasSgemm`
  (size 4096). Should pass once your TODOs are filled in.
- `bench/bench_gemm.cpp` — provided. Microbenchmarks all four
  versions and `cublasSgemm`, prints GFLOP/s and % of cuBLAS. Used
  for the §3 perf target.
- `python/gemm_ext.py` — provided. JIT loader (`cpp_extension.load`)
  + thin Python wrapper. **Do not modify.**
- `python/test_gemm.py` — provided. Two tests: numerics vs
  `torch.matmul` (sizes 128, 512, 1024, 4096) and a wrapper-overhead
  bound (< 5% of kernel time at 4096³, comparing wrapped op to a
  raw CUDA-event timing of the underlying kernel).
- `python/README.md` — provided. How to run.

**Definition of done.**
1. `cmake -S . -B build && cmake --build build && ctest --test-dir build`
   all pass.
2. `pytest python/` passes (numerics + overhead bound).
3. `bench/bench_gemm` reports `gemm_tiled_async` ≥ 50% of
   `cublasSgemm` at 4096³.
4. `report/` contains a Nsight Compute `.ncu-rep` for each of the
   four versions, named `ncu_gemm_v{0,1,2,3}.ncu-rep`.

## 5. Hypothesis

*Write this BEFORE you start optimizing.* Predict, for each version:

- **v0 (naive)**: bottleneck is global-memory bandwidth, arithmetic
  intensity is ~1 FLOP per byte loaded → severely memory-bound. Expect
  Nsight Compute to show ~85% memory throughput, ~5-10% compute
  throughput, and < 10% of cuBLAS.
- **v1 (32×32 tiled)**: each `A` and `B` element loaded from global
  memory is reused 32 times in shared memory → arithmetic intensity
  jumps ~32×. Expect 2-4× speedup, but bank conflicts on the 32-wide
  tile leave ~30% on the table.
- **v2 (64×64 padded, register-tiled)**: padding eliminates bank
  conflicts and the 4×4 register tile collapses the inner loop. Expect
  ~30-50% of cuBLAS.
- **v3 (async copies)**: overlapping the next tile's global→shared
  copy with the current tile's MMAs hides ~all of the L2→shared
  latency. Expect to clear the 50% bar.

TODO: fill in the predicted GFLOP/s before you measure.

## 6. Method

Implement and benchmark in this order. Do **not** skip ahead.

### v0 — naive (one thread = one output element)
Each thread computes one `C[row, col]` by looping over `k`. Reads
`A[row, k]` and `B[k, col]` from global memory `K=4096` times. This
is the strawman that proves why we need tiling.

### v1 — 32×32 shared-memory tile
Block of 32×32 threads cooperatively loads a 32×32 tile of `A` and
`B` into shared memory, then each thread accumulates into one `C`
element across the tile. Outer loop walks tiles along `K`.
`__syncthreads()` between load and use, and between use and next
load. Bank conflicts will hurt — measure them in Nsight Compute,
don't fix them yet.

### v2 — 64×64 tile, padded, 4×4 register tile per thread
256 threads per block, each computes a 4×4 sub-tile of `C` in
registers. Shared-memory tile is `64 × (64+P)` for the padding `P`
that eliminates 32-way bank conflicts (you choose `P` — see TODO).
Vectorize the global loads with `float4` if you have time.

### v3 — async copies + pipeline (the perf target)
Same shape as v2, but uses `cuda::memcpy_async` + `cuda::pipeline`
with **two-stage double buffering**: while threads multiply the
current shared tile, the next tile is being copied from global into
the *other* shared buffer. This is the technique that closes the gap
to cuBLAS at this size.

## 7. Results

Fill in after you run `bench/bench_gemm` on Spark.

| Version              | Time (ms) | GFLOP/s | % of cuBLAS |
|----------------------|-----------|---------|-------------|
| v0_naive             |           |         |             |
| v1_tiled32           |           |         |             |
| v2_tiled64_padded    |           |         |             |
| v3_tiled_async       |           |         |             |
| `cublasSgemm` (ref)  |           | —       | 100%        |

Reference Nsight reports:
- `report/ncu_gemm_v0.ncu-rep`
- `report/ncu_gemm_v1.ncu-rep`
- `report/ncu_gemm_v2.ncu-rep`
- `report/ncu_gemm_v3.ncu-rep`

Python wrapper overhead at M=N=K=4096:
- Wrapped op: ___ ms/call
- Raw kernel: ___ ms/call
- Overhead: ___ % (target < 5%)

## 8. Discussion

For each version, write one paragraph answering:

- Did the predicted bottleneck match what Nsight showed?
- What was the single biggest source of stalls (per "Warp State
  Statistics")?
- What would you change in your hypothesis next time?

For v3 specifically: did `cuda::memcpy_async` actually overlap
copy and compute, or did the pipeline structure introduce a stall
you didn't expect? Cite the Nsight section.

TODO

## 9. References

- PMPP 4e §4 (Compute architecture), §5 (Memory architecture and
  data locality) — primary text for this week.
- CUDA C++ Best Practices Guide §9 (Memory optimizations).
- CUDA C++ Programming Guide §B.7 (`cuda::memcpy_async`,
  `cuda::pipeline`).
- Iglberger, *C++ Software Design* Ch 4-5.
- Curriculum: `.cursor/skills/curriculum-plan/month-1-foundations.md`
  (Lab 02 / Week 2).
- Skills: `.cursor/skills/cuda-kernel-authoring/SKILL.md`,
  `.cursor/skills/nsight-profiling/SKILL.md`,
  `.cursor/skills/python-bindings/SKILL.md`.

## 10. What I would do next

One paragraph. Candidate stretch directions for Lab 04's
checkpoint: vectorize all global loads as `float4`, double the
register-tile to 8×8 per thread, sweep block shapes (64×64, 128×64,
128×128), or — the Blackwell move — start prototyping a TMA-based
copy path. Pick one and explain why you'd attempt it first.

TODO
