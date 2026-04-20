# Week 03 — Reductions, Scans, Atomics, Warp Primitives  (Tier A)

## 0. Intro

Most non-GEMM GPU workloads — softmax denominators, prefix-sum
boundaries in segmentation, histogram equalization, integral images
for spatial features — are some flavor of *reduce*, *scan*, or
*scatter*. This week you build the canonical primitives from scratch,
in the same five stages Mark Harris used in his classic 2007 NVIDIA
talk, and you compare them line-by-line against `cub::DeviceReduce` /
`cub::DeviceScan`. By the end of the week you will know exactly why
warp shuffles exist and why `__syncthreads()` is more expensive than
it looks. You also wrap the kernels as PyTorch custom ops so the
Month-4 application stack can call them.

> **New terms this week.** See [`GLOSSARY.md`](./GLOSSARY.md) in this
> folder. Read it before §3 Spec — it covers the CUDA, C++20, and
> Python-bindings terms that appear here for the first time. (No new
> CV / Spatial Intelligence terms; that vocabulary starts in Month 3.)

## 1. What you will learn

By the end of this lab you will be able to:

### CUDA
- Walk through Mark Harris's five reduction stages and explain, with
  Nsight evidence, what bottleneck each stage removes (warp
  divergence → bank conflicts → idle threads at load → instruction
  count via warp shuffles).
- Use `__shfl_down_sync` and `__shfl_xor_sync` to do an entire warp
  reduction with no shared memory and no `__syncthreads`.
- Use `cooperative_groups` (`thread_block`, `tiled_partition<32>`,
  `inclusive_scan`) as a higher-level alternative to raw shuffles.
- Privatize a histogram in shared memory and use warp aggregation
  (`__match_any_sync` / `__ballot_sync`) to collapse N atomics into 1.
- Write a single-pass grid-stride reduction whose `gridDim.x` is
  tuned for occupancy on Spark `sm_121`, not for `n`.

### C++20
- Use `enum class` + a switch in the launcher as the Iglberger Ch 5
  Strategy / dispatch pattern (same idea as `gemm.hpp`'s
  `GemmVersion`, applied to three primitive families).
- Use C++20 lambdas as the unit of microbench timing (the bench file
  takes `auto&& fn` and times any callable).

### CV / Spatial Intelligence
- *N/A this week — these primitives are CV-adjacent (histograms,
  prefix sums for integral images, scan-based RLE) but the concrete
  spatial-intelligence applications start in Month 3.*

### Python bindings
- Wrap *three* primitive families behind one extension module via
  `torch.utils.cpp_extension.load()`, exposing each as a versioned
  PyTorch op.
- Verify numerics from Python against `torch.sum`, `torch.cumsum`,
  and `torch.bincount`.
- Bound wrapper overhead (Tier-A coarse rule this week; the strict
  5%-of-kernel-time rule is enforced from Week 5 once the C++ bench
  writes a JSON file the test can read).

## 2. Prerequisites

**Readings (do these first):**
- PMPP 4e Ch 6 (Performance considerations), Ch 9 (Parallel
  histogram, atomics), Ch 10 (Reduction), Ch 11 (Prefix sum / scan).
- Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA 2007
  whitepaper / slides). Still required reading.
- CUDA C++ Programming Guide §B.16 (warp shuffle), §B.18
  (cooperative groups).
- CUB docs: `DeviceReduce::Sum`, `DeviceScan::InclusiveSum`.

**Prior weeks' deliverables you need working:**
- Week 01: `DeviceBuffer<T>`, `Stream`, `KernelLaunch` helpers (we
  re-use the same launch idioms here).
- Week 02: comfort with `__syncthreads`, shared memory, the
  `enum class` + dispatcher pattern.

**Toolchain checks:**
- `nvidia-smi`, `nvcc --version`, CMake ≥ 3.28 (see
  [`docs/GETTING-STARTED.md`](../../docs/GETTING-STARTED.md) if
  anything is missing). CUB ships with the CUDA Toolkit; no extra
  install needed.

## 3. Spec

### Inputs / outputs / dtypes

- **Reduction.** `float* d_in` of length `n` → single `float` at `*d_out`.
  `dtype = float32`. Sum is the binary op (other ops are stretch).
- **Scan.** `float* d_in` length `n ≤ 1024` (single-block this week)
  → `float* d_out` of the same length, inclusive prefix sum.
- **Histogram.** `uint8_t* d_in` length `n` → `uint32_t* d_bins`
  length 256 (256-bin histogram of the byte stream).

### Numerical tolerance

- Reduction: `|gpu - cpu| ≤ max(1e-2, 1e-4 * |cpu|)` for `n ≤ 1<<22`.
  (Float summation is non-associative; tighten via Kahan in §10
  stretch, do not tighten the test.)
- Scan: `max_i |gpu[i] - cpu[i]| ≤ 1e-2` at `n = 1024`.
- Histogram: bitwise equality with `torch.bincount`.

### Performance target

Measured on Spark `sm_121` at the bench sizes below.

- `reduce_v4 / cub::DeviceReduce::Sum  ≥ 80%` at `n = 2^28`.
- `scan_best  / cub::DeviceScan::InclusiveSum  ≥ 70%` at `n = 1024`.
- Histogram: `shared_warp` strictly faster than `global` on
  uniform-random `uint8_t` input at `n = 2^26`. (Tighter target lands
  in Week 4 once we benchmark on a real image stream.)
- Python wrapper overhead: coarse Tier-A rule — `reduce("v4")` within
  30% of `reduce_cub()` Python-side. The strict 5% rule lands in W5.

## 4. Your task

This week is **Tier A**. See
[`labs/_template/README.md`](../_template/README.md) for what that
means: every file already compiles and runs; the `// TODO(student):`
markers point at the specific lines that turn the lab from "compiles"
into "passes the perf target". Concretely:

- `src/reduce.hpp` — fully provided. Defines `ReduceVersion` and the
  two launcher prototypes. Do not edit.
- `src/reduce.cu` — five kernels (v0..v4). The kernel skeletons
  compile and produce correct results. **Three TODO(student):
  markers**:
    1. `reduce_v0`: write the divergent inner-loop body so v0 is
       observably slower than v1 in the bench.
    2. `reduce_v2`: write the reverse-stride, sequential-addressing
       inner loop. This is the Harris stage-3 trick.
    3. `warp_reduce_sum`: replace the loop with five explicit
       `__shfl_down_sync` calls (offsets 16, 8, 4, 2, 1). Idiom-axis
       grading lives here.
   Plus a tuning TODO on the v4 launch config — sweep
   `grid ∈ {256, 512, 1024, 2048}` and report the winner in §7.
- `src/reduce_cub.cu` — fully provided. CUB baseline.
- `src/scan.hpp` — fully provided.
- `src/scan.cu` — both scan kernels compile and produce correct
  results. **Two TODO(student): markers**:
    1. `scan_hillis_steele`: confirm/implement the ping-pong step and
       explain in §6 why the work is O(n log n) vs Brent-Kung's O(n).
    2. `scan_coop_groups`: pick the partition (whole block vs
       per-warp `tiled_partition<32>` + reassembly) that wins on
       Spark, and show the delta in §7.
- `src/scan_cub.cu` — fully provided.
- `src/histogram.hpp` — fully provided.
- `src/histogram.cu` — both kernels compile and produce correct
  results. **One TODO(student) marker**: replace the
  shared-memory `atomicAdd`-per-element with the warp-aggregated
  variant (`__ballot_sync` + `__match_any_sync` + one `atomicAdd`
  per group). This is the whole point of the privatized version.
- `src/reduce_scan_pybind.cpp` — fully provided. Exposes `reduce`,
  `reduce_cub`, `scan`, `scan_cub`, `histogram` as Python ops.
- `tests/test_reduce_scan.cpp` — fully provided. GoogleTest with
  parametric `ReduceVersion` / `ScanVersion` cases plus a histogram
  test. Should pass at green/scaffold time except for whatever your
  TODOs break — that's the point.
- `bench/bench_reduce_scan.cpp` — fully provided. Reports best-of-K
  ms, GB/s, and `% of CUB` for each version.
- `python/reduce_scan_ext.py` — fully provided. JIT loader + thin
  wrappers + smoke test.
- `python/test_reduce_scan.py` — fully provided. pytest with
  numerics vs `torch.{sum,cumsum,bincount}` and a coarse
  wrapper-overhead bound.

**Definition of done.** `ctest --test-dir build -V` is all green;
`pytest python/` is all green; `bench_reduce_scan` reports
`v4 ≥ 80% of cub::DeviceReduce` and `scan_best ≥ 70% of
cub::DeviceScan`; `report/` contains a Nsight Systems trace and at
least one Nsight Compute report for `reduce_v4` showing the warp
shuffle in the SASS.

## 5. Hypothesis

*Write this BEFORE you start coding the optimized version.*

Predict, for each version, the dominant bottleneck Nsight Compute will
show:

- v0 — warp divergence (Stall: Branch).
- v1 — bank conflicts (Memory Workload Analysis: shared bank
  conflicts > 0).
- v2 — `__syncthreads` cost (Stall: Barrier dominates).
- v3 — half the launches but still `__syncthreads`-bound at the tail.
- v4 — bandwidth-bound (Speed of Light: Memory ≫ Compute).

Predict the achieved throughput as a fraction of CUB and write it
down. The §8 Discussion will be graded against this prediction, not
against the final number alone.

TODO

## 6. Method

### v0 — interleaved addressing, divergent warps
Harris stage 1. `if (tid % (2*s) == 0)` makes half the warp idle at
the first iteration, a quarter at the second, etc.

### v1 — interleaved addressing, strided index
Harris stage 2. Same arithmetic, contiguous active threads.

### v2 — sequential addressing
Harris stage 3. Reverse the stride; eliminates bank conflicts.

### v3 — first add during load
Harris stage 4. Each block now covers `2 * BLOCK` elements; halves
launch count.

### v4 — warp shuffle + grid-stride
Harris stage 7 + grid-stride loop. Single-pass: each thread accumulates
many input elements, then a warp shuffle collapses 32 → 1, then one
warp folds the per-warp sums, then `atomicAdd` into the output.

### Scan v_hs / v_cg
Two single-block scans — Hillis-Steele (O(n log n) work, O(log n)
span) and `cooperative_groups::inclusive_scan` (built on shuffles
internally). Compare both to `cub::DeviceScan::InclusiveSum`.

### Histogram v_g / v_sw
Two histograms — global atomics, vs shared-memory privatization with
warp aggregation. Compare both to `torch.bincount` for correctness.

## 7. Results

| Kernel                        | Time (ms) | GB/s | % of CUB |
|---|---|---|---|
| `cub::DeviceReduce::Sum`      |     |     | 100%     |
| `reduce_v0`                   |     |     |          |
| `reduce_v1`                   |     |     |          |
| `reduce_v2`                   |     |     |          |
| `reduce_v3`                   |     |     |          |
| `reduce_v4` (target ≥ 80%)    |     |     |          |
| `cub::DeviceScan::InclusiveSum` |     |  -  | 100%     |
| `scan_hillis_steele`          |     |  -  |          |
| `scan_coop_groups` (target ≥ 70%) |     |  -  |          |
| `hist_global`                 |     |     |    -     |
| `hist_shared_warp`            |     |     |    -     |

Reference Nsight reports (commit these to `report/`):
- `report/nsys_reduce_all.qdrep`
- `report/ncu_reduce_v4.ncu-rep`
- `report/ncu_hist_shared_warp.ncu-rep`

## 8. Discussion

Did the predicted bottleneck show up for each version? Cite the
Nsight Compute section and the specific metric. What surprised you?
What would you change in your hypothesis next time? Specifically:

- Which Harris stage gave the biggest single-step speedup?
- Did `cooperative_groups::inclusive_scan` beat your hand-rolled
  Hillis-Steele? By how much, and why?
- For the histogram, what was the input distribution's effect on the
  warp-aggregation win? (Skewed → bigger win; uniform → smaller win.)

TODO

## 9. References

- PMPP 4e §6 (Performance considerations), §9 (Parallel histogram),
  §10 (Reduction), §11 (Prefix sum).
- Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA 2007).
- CUDA C++ Programming Guide §B.16 (warp shuffle), §B.18
  (cooperative groups).
- CUB docs: `DeviceReduce`, `DeviceScan`, `WarpReduce`.
- Iglberger Ch 5 (Strategy modernization).
- Curriculum: `.cursor/skills/curriculum-plan/month-1-foundations.md`
  — Week 03.
- Skills: `.cursor/skills/cuda-kernel-authoring/SKILL.md`,
  `.cursor/skills/nsight-profiling/SKILL.md`,
  `.cursor/skills/python-bindings/SKILL.md`.

## 10. What I would do next

Implement Brent-Kung scan (O(n) work) and benchmark against
Hillis-Steele and CUB. Then implement Merrill-Garland decoupled
look-back so the scan generalizes past `n = 1024`. Add Kahan
compensated summation to `reduce_v4` and quantify the accuracy gain
on adversarial inputs (alternating large/small magnitudes).

TODO
