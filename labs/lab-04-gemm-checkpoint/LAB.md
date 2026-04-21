# Lab 04 — Checkpoint: production-grade tiled GEMM

> Month-1 checkpoint week. The bar is **17/20**, not 14. This is the
> first week the curriculum will refuse to advance you on a "good
> enough" result. Read [`labs/_template/README.md`](../_template/README.md)
> for what Tier A scaffolding means, and re-read your Week 02 lab
> before you touch this one.

## 0. Intro

The single point of this week is to show that you can drive **one
kernel** to a real performance target with **profile evidence** to
back it up. You will start from your best Week-02 GEMM
(`gemm_tiled_async`), add the four optimizations the literature says
matter most for SGEMM on a modern NVIDIA GPU — double-buffered async
copies, register tiling (each thread owns a 4×4 sub-tile of `C`),
vectorized `float4` loads, and read-only `__ldg` hints — sweep tile
sizes (32 / 64 / 128), and benchmark against both `cublasSgemm` and a
CUTLASS-template GEMM you wire up. The checkpoint deliverable is a
paper-style `report/LAB.md` with a roofline diagram, an Nsight Compute
report per variant, and a one-paragraph diagnosis of why each
optimization moved (or failed to move) the needle. Pass criterion:
**≥ 70% of `cublasSgemm`** at M=N=K=4096 single precision; **≥ 80%**
earns the bonus point on the Performance axis.

> **New terms this week.** See [`GLOSSARY.md`](./GLOSSARY.md) in this
> folder. Read it before §3 Spec — it covers the CUDA, C++20, CV /
> Spatial Intelligence, and Python-bindings terms that are introduced
> for the first time this week.

## Plan of work — order of operations

This is a **checkpoint lab** — the bar is **17/20**, not 14, and
`/checkpoint` will refuse to advance you on a "good enough" result.
Work top-to-bottom; don't move past a checkbox until it's green.

### A. Read first (do not skip)

- [ ] PMPP 4e Ch 5 (Memory architecture and data locality) — yes,
      again.
- [ ] PMPP 4e Ch 6 (Performance considerations) — yes, again.
- [ ] CUDA C++ Best Practices Guide §9 (Memory optimizations)
      end-to-end.
- [ ] CUTLASS, *Efficient GEMM in CUDA* — vocabulary (warp tiles,
      MMA fragments, double buffering, thread-block / warp / MMA
      tile hierarchy) is required for §5 Hypothesis and §8
      Discussion.
- [ ] Williams, Waterman, Patterson — *Roofline: An Insightful
      Visual Performance Model* (CACM 2009). You will draw one in
      §7.
- [ ] Skim this lab's [`GLOSSARY.md`](./GLOSSARY.md).

### B. Bring the scaffold up on Spark

- [ ] Confirm Lab 02 is green and `gemm_tiled_async` ≥ 50% of
      `cublasSgemm`. **If Lab 02 isn't green, fix Lab 02 first.**
- [ ] Toolchain: CUDA 13, CMake ≥ 3.28, Ninja, Nsight on `PATH`,
      torch (CUDA 13 / sm_121) + pytest available.
- [ ] CUTLASS available (`CUTLASS_DIR` set or installed where
      CMake's `find_package` will find it). Without it the CUTLASS
      baseline target is silently disabled and you lose 1
      Performance point — fix this.
- [ ] `cmake -S . -B build -G Ninja && cmake --build build -j`.
- [ ] Baseline `ctest --test-dir build --output-on-failure`. Slow
      reference paths pass; the optimized path fails until Phase D.

### C. Write §5 Hypothesis *before* you optimize

- [ ] State Lab-02's `v3_tiled_async` Speed-of-Light bottleneck at
      4096³ from its Nsight Compute report — memory- or
      compute-bound, with the SOL number.
- [ ] Predict, in order, which of the four optimizations
      (register tile → `float4` → `__ldg` → double-buffer) moves
      the needle the most, and which Nsight counter (DRAM
      throughput, L1 hit rate, `smsp__inst_executed_pipe_fma`,
      warp-stall reason) will confirm it.
- [ ] Predict the achieved fraction of `cublasSgemm`. Commit to §5.

### D. Implement the §4 TODOs and turn ctest green

- [ ] `src/gemm_v4_checkpoint.cu` — (1) `float4` global→shared load
      for `A`, (2) `float4 + __ldg` global→shared load for `B`,
      (3) 4×4 register-tile inner loop (`a_reg[4]`, `b_reg[4]`,
      16 FMAs per K-step), (4) second shared-memory buffer +
      `cuda::pipeline<...>::producer_acquire/commit` swap for
      double buffering.
- [ ] `src/gemm_launch.cu` — pick `(grid, block)` for the 128-tile
      `if constexpr` instantiation.
- [ ] `src/cutlass_baseline.cu` — pick the
      `cutlass::gemm::device::Gemm<...>` template parameters
      (RowMajor, `float`, SIMT, tile shape) so it compiles against
      your CUTLASS.
- [ ] `python/gemm_checkpoint_ext.py` — extend `_VERSION_MAP` with
      `"v4_checkpoint": 4` and add a smoke-test call.
- [ ] `ctest -R gemm_checkpoint` is green at all sizes.

### E. Get the Python bindings green

- [ ] `pytest python/test_gemm_checkpoint.py` — numerics vs
      `torch.matmul` and the **< 5%** wrapper-overhead bound at
      M=N=K=4096.

### F. Run the bench, sweep tiles, hit the perf target

- [ ] `./build/bench_gemm_checkpoint` — captures **median over
      20 runs after 5 warm-ups** for each variant + tile size.
- [ ] Re-bench `v3_tiled_async` (carry-forward) on this harness so
      §7 starts from a known baseline.
- [ ] Bench v4a / v4b / v4c / v4_checkpoint at BM ∈ {32, 64, 128}.
- [ ] **Pass: `v4_checkpoint` ≥ 70% of `cublasSgemm`** at
      M=N=K=4096 single precision.
- [ ] **Bonus: `v4_checkpoint` ≥ 80% of `cublasSgemm`** earns the
      Performance bonus point.
- [ ] CUTLASS baseline runs and reports a number (not used as the
      pass threshold, but must build).

### G. Profile (mandatory at checkpoint rigor)

Follow `.cursor/skills/nsight-profiling/SKILL.md`. Commit raw
artifacts under `report/`.

- [ ] `report/ncu_v4_checkpoint.ncu-rep` — **full sections**, not
      just SOL. Include Memory Workload Analysis, Compute
      Workload Analysis, Source Counters, Warp State Statistics.
- [ ] `report/nsys_v4_checkpoint.qdrep` — one bench run.
- [ ] `report/roofline.png` (or `.svg`) — plot v3_tiled_async, v4a,
      v4b, v4c, v4_checkpoint, `cublasSgemm`, and CUTLASS on the
      Spark roofline.

### H. Write the paper-style report

- [ ] §7 Results table — fill **every** row.
- [ ] `report/LAB.md` — paper-style writeup with the roofline diagram
      embedded, one paragraph of Nsight diagnosis per optimization
      attributing speedup to a specific counter, and a final
      results table.
- [ ] §8 Discussion — for each optimization, did the predicted
      counter actually move and by how much? Where on the roofline
      does `v4_checkpoint` sit relative to `cublasSgemm`? Is the
      remaining gap arithmetic, scheduling, or memory? If you
      missed 70%, what is the *minimum* change that would close the
      gap, and why didn't you make it?
- [ ] §10 What I would do next — pick one of: tensor-core WMMA
      route (on-ramp to Lab 05) or non-multiple-of-128 tail loop.
      One paragraph. Not both.
- [ ] Run `/lab-report` to polish.

### I. Self-grade (strict — checkpoint rigor)

- [ ] `/checkpoint` against the 5-axis rubric in
      `.cursor/skills/weekly-checkpoint/SKILL.md`. **≥ 17/20** to
      advance — not 14. Below 17 means rework, not retry-with-vibes.

### Definition of done for Lab 04

`ctest -R gemm_checkpoint` is green; `pytest
python/test_gemm_checkpoint.py` is green; `bench_gemm_checkpoint`
shows `v4_checkpoint ≥ 70%` of `cublasSgemm` at M=N=K=4096 with
`< 5%` Python wrapper overhead; `report/` holds
`ncu_v4_checkpoint.ncu-rep` (full sections),
`nsys_v4_checkpoint.qdrep`, `roofline.png`, and a paper-style
`LAB.md`; this file's §5, §7, §8, §10 are written;
`/checkpoint` returns ≥ 17/20.

## 1. What you will learn

By the end of this lab you will be able to:

### CUDA
- Combine four canonical SGEMM optimizations on the same kernel and
  attribute the speedup of each to a Nsight Compute counter
  (double-buffered `cuda::memcpy_async`, register tiling, `float4`
  vectorized loads, `__ldg` read-only cache hints).
- Read a roofline plot and place each variant on it: where the kernel
  is memory-bound, where it crosses to compute-bound, and what the
  remaining gap to `cublasSgemm` is *bounded by*.
- Sweep launch geometry (BM × BN × BK = 32 / 64 / 128) and reason
  about the occupancy / register-pressure / shared-memory trade-off
  rather than guess.
- Wire a CUTLASS GEMM from a provided template and use it as a
  second baseline alongside `cublasSgemm`.

### C++20
- Extend the Week-02 strategy enum (`GemmVersion`) with a new
  `v4_checkpoint` variant without breaking the existing dispatcher
  contract — Iglberger Ch 5 in practice.
- Express a tile-size sweep with `if constexpr` over a small set of
  compile-time tile parameters so each tile size compiles to its own
  register-allocated kernel.

### CV / Spatial Intelligence
*N/A this week — Month 1 is pure C++20 + CUDA foundations.*
GEMM is the substrate every Month-3 spatial-intelligence model
(NeRF MLPs, Gaussian-splat projection, attention) eventually leans
on, so the perf number you ship here is what those weeks will
inherit.

### Python bindings
- Re-expose the new `v4_checkpoint` kernel through the same JIT
  `cpp_extension.load()` wrapper your Week-02 lab uses (Pattern A
  from `.cursor/skills/python-bindings/SKILL.md`).
- Verify Python-side numerics against `torch.matmul` to FP32
  tolerance, and assert wrapper overhead **< 5%** of kernel time at
  M=N=K=4096.

## 2. Prerequisites

**Readings (do these first):**
- PMPP 4e Ch 5 (Memory architecture and data locality) — yes, again.
- PMPP 4e Ch 6 (Performance considerations) — yes, again.
- CUDA C++ Best Practices Guide §9 (Memory optimizations) end-to-end.
- CUTLASS docs — *Efficient GEMM in CUDA*
  (https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md).
  You will *not* use CUTLASS as your primary kernel until Week 5,
  but the vocabulary (warp tiles, MMA fragments, double buffering,
  thread-block tile / warp tile / MMA tile hierarchy) is required
  for the §5 Hypothesis and §8 Discussion.

**Prior weeks' deliverables you need working:**
- Week 02: `gemm_tiled_async` passes `test_gemm` and reports a
  GFLOP/s number that meets its ≥ 50% target. If Week 02 isn't
  green, do **not** start Week 04 — fix Week 02 first.
- Week 01: `DeviceBuffer<T>`, `Stream`, and the `KernelLaunch`
  helper. They are reused unchanged here.
- Week 03: optional. `cub::DeviceReduce` is *not* used this week.

**Toolchain checks:**
- `nvidia-smi`, `nvcc --version` (CUDA 13), CMake ≥ 3.28, Nsight
  Systems + Nsight Compute on `PATH`. See
  [`docs/GETTING-STARTED.md`](../../docs/GETTING-STARTED.md) and the
  [`dgx-spark-setup`](../../.cursor/skills/dgx-spark-setup/SKILL.md)
  skill if anything is missing.
- For the Python wrapper: a virtualenv with `torch`
  (CUDA 13 / sm_121-compatible build) and `pytest`.
- For the CUTLASS baseline: CUTLASS source on `CUTLASS_DIR` (the
  CMake script will `find_package` it; if absent, the CUTLASS
  baseline target is silently disabled and you lose 1 Performance
  point — fix this).

## 3. Spec

This is the contract. §4 Your task tells you which parts of it you
implement at Tier A.

### Inputs / outputs / dtypes
Single-precision row-major SGEMM:

```
C = alpha * A @ B + beta * C
```

- `A` is `M x K`, `float`, row-major, device-resident.
- `B` is `K x N`, `float`, row-major, device-resident.
- `C` is `M x N`, `float`, row-major, device-resident, may be the
  same buffer on input and output.
- `alpha`, `beta` are host-side `float`.
- All shapes assumed multiples of the largest tile (128) for the
  checkpoint measurement; non-multiple shapes are a stretch goal in
  §10.

### Numerical tolerance
Max-abs-error vs a CPU reference (or `cublasSgemm`) at M=N=K=1024:

- `v4_checkpoint`: `max_abs_err / (K * max|A| * max|B|) < 1e-5`.

This is a relative bound because the absolute error grows with `K`;
SGEMM at K=4096 is not bounded by single-ulp accuracy and the lab
is graded as such.

### Performance target
At M=N=K=4096, single precision, on Spark sm_121, **median over
20 runs after 5 warm-up runs**:

- **Pass:** `v4_checkpoint` ≥ **70%** of `cublasSgemm` GFLOP/s.
- **Bonus:** `v4_checkpoint` ≥ **80%** of `cublasSgemm` GFLOP/s.
- CUTLASS baseline must build and run; its number is reported but
  not used as the pass threshold.
- Python wrapper overhead < **5%** of kernel time at M=N=K=4096
  (measured: `(t_python - t_cuda_event) / t_cuda_event`).

## 4. Your task

This week is **Tier A** — *show + small fill-in*. The dispatcher,
launcher, RAII, tests, bench harness, and Python wrapper compile and
run as shipped. You implement the parts marked
`// TODO(student):` (2-4 per file).

See [`labs/_template/README.md`](../_template/README.md) for the
tier rules.

Concretely:

- `src/gemm.hpp` — **provided.** Extends Week-02's `GemmVersion`
  with `v4_checkpoint`. Do not change the public ABI.
- `src/gemm_v4_checkpoint.cu` — **the kernel you ship this week.**
  Compiles as-is with a correctness-passing but slow inner loop.
  Four `// TODO(student):` markers:
  1. Replace the scalar global load of `A` with a `float4` load
     into shared memory.
  2. Replace the scalar global load of `B` with a `float4` +
     `__ldg` load into shared memory.
  3. Implement the 4×4 register tile inner loop (two thread-local
     `float a_reg[4]` / `b_reg[4]` arrays + 16 FMAs per K step).
  4. Add the second shared-memory buffer + the
     `cuda::pipeline<...>::producer_acquire/commit` swap that
     turns the loop into a double-buffered pipeline.
- `src/gemm_launch.cu` — **provided.** Routes `v4_checkpoint` to
  the new kernel and sweeps `BM/BN/BK ∈ {32, 64, 128}` via
  `if constexpr` over a `TileConfig` non-type template parameter.
  One `// TODO(student):` marker: pick the launch geometry
  (`grid`, `block`) for the 128-tile instantiation.
- `src/gemm_pybind.cpp` — **provided.** Adds `v4_checkpoint` to
  the `_VERSION_MAP` already exported by Week 02. No TODOs.
- `src/cutlass_baseline.cu` — **provided** (guarded by
  `CUDALAB_HAVE_CUTLASS`). One `// TODO(student):` marker: pick
  the `cutlass::gemm::device::Gemm<...>` template parameters
  (layout = `RowMajor`, element types = `float`, op class = SIMT,
  tile shape) so it actually compiles against your installed
  CUTLASS.
- `tests/test_gemm_checkpoint.cpp` — **provided.** GoogleTest with
  CPU reference (small N) + `cublasSgemm` reference (large N).
  No TODOs.
- `bench/bench_gemm_checkpoint.cpp` — **provided.** Sweeps tile
  sizes, prints a markdown table, computes `% of cuBLAS`. No TODOs.
- `python/gemm_checkpoint_ext.py` — **provided.** JIT loader +
  thin wrapper exposing `gemm(..., version="v4_checkpoint")`.
  One `# TODO(student):` marker: extend `_VERSION_MAP` with
  `"v4_checkpoint": 4` and add a smoke-test call at the bottom.
- `python/test_gemm_checkpoint.py` — **provided.** `pytest`s for
  numerics vs `torch.matmul` and the < 5% overhead bound. No TODOs.
- `report/LAB.md` — **you write.** Paper-style writeup with
  roofline diagram (PNG / SVG checked in), one paragraph of Nsight
  diagnosis per optimization, and the final results table from
  `bench_gemm_checkpoint`.

**Definition of done.**

1. `cmake --build build --target test_gemm_checkpoint && ctest -R gemm_checkpoint`
   passes.
2. `pytest python/test_gemm_checkpoint.py` passes.
3. `bench_gemm_checkpoint` reports `v4_checkpoint` ≥ 70% of
   `cublasSgemm` at M=N=K=4096.
4. `report/` contains:
   - `ncu_v4_checkpoint.ncu-rep` (full sections, not just SOL).
   - `nsys_v4_checkpoint.qdrep` (one bench run).
   - `roofline.png` (or `.svg`).
   - `LAB.md` (paper-style writeup).

## 5. Hypothesis

*Write this BEFORE you start coding the optimized version.*

What is the bottleneck you predict for `v3_tiled_async` at
M=N=K=4096 — memory or compute? What does the Week-02 Nsight
Compute SOL section say? What fraction of `cublasSgemm` GFLOP/s
do you expect each optimization to recover, and in what order will
you stack them? Specifically, which of the four optimizations do
you predict moves the needle the *most*, and what counter (DRAM
throughput? L1 hit rate? `smsp__inst_executed_pipe_fma`?
warp-stall reasons?) will tell you so?

TODO

## 6. Method

### v3_tiled_async (Week-02 carry-forward)
Re-bench the Week-02 winner unchanged on the new harness so this
week's table starts from a known number. No code change.

### v4a — register tiling only
Same shared-memory layout as `v3_tiled_async`, but each thread now
owns a 4×4 sub-tile of `C` in registers. Expect L1/shared traffic
to drop by ~4×; expect register pressure to bite occupancy.

### v4b — v4a + `float4` vectorized loads
Replace scalar global loads of `A` and `B` with 16-byte `float4`
loads into shared memory. Expect DRAM transactions per kernel to
roughly halve.

### v4c — v4b + `__ldg` (read-only cache)
Mark the global loads of `A` and `B` as read-only via `__ldg` (or
the equivalent `__restrict__ const` + nvcc heuristic). Expect a
modest L1 hit-rate bump.

### v4_checkpoint — v4c + double buffering
Add a second shared-memory tile and a `cuda::pipeline<scope, 2>`
so tile `k+1` is being copied while tile `k` is being multiplied.
Expect *latency hiding* — the kernel approaches the DRAM roofline.

### Tile-size sweep
For `v4_checkpoint`, instantiate `BM × BN × BK ∈ {32, 64, 128}`
via `if constexpr` and report all three. Pick a winner; explain.

### CUTLASS baseline
Wire one CUTLASS SIMT SGEMM template; report its number alongside
`cublasSgemm`. Don't tune it — this is a sanity baseline, not a
competition.

## 7. Results

| Variant | Time (ms) | GFLOP/s | % of cuBLAS | Notes |
|---|---|---|---|---|
| `cublasSgemm` (baseline) |  |  | 100% | reference |
| CUTLASS SIMT (template defaults) |  |  |  | sanity baseline |
| `v3_tiled_async` (Week-02 carry-forward) |  |  |  | starting point |
| `v4a` register-tile only |  |  |  |  |
| `v4b` + `float4` loads |  |  |  |  |
| `v4c` + `__ldg` |  |  |  |  |
| `v4_checkpoint` (BM=64) |  |  |  |  |
| `v4_checkpoint` (BM=128) |  |  |  |  |

**Python wrapper overhead at M=N=K=4096:** TODO % (target < 5%).

Reference Nsight reports:
- `report/ncu_v4_checkpoint.ncu-rep`
- `report/nsys_v4_checkpoint.qdrep`
- `report/roofline.png`

## 8. Discussion

For each optimization (register tile → `float4` → `__ldg` →
double-buffer), did the predicted Nsight counter actually move,
and by how much? Where is your `v4_checkpoint` on the roofline
relative to `cublasSgemm`? Is the remaining gap arithmetic
(missing tensor-core path), scheduling (occupancy / register
spills), or memory (L2 hit-rate ceiling)? If you missed 70%, what
is the *minimum* change that would close the gap, and why didn't
you make it?

TODO

## 9. References

- PMPP 4e §5 (Memory architecture), §6 (Performance considerations).
- CUDA C++ Best Practices Guide §9.
- CUTLASS, *Efficient GEMM in CUDA*
  (https://github.com/NVIDIA/cutlass).
- libcu++ `<cuda/pipeline>`, `<cuda/barrier>`.
- Williams, Waterman, Patterson — *Roofline: An Insightful Visual
  Performance Model*, CACM 2009.
- Curriculum: `.cursor/skills/curriculum-plan/month-1-foundations.md` §Week 4.
- Skills: `.cursor/skills/cuda-kernel-authoring/SKILL.md`,
  `.cursor/skills/nsight-profiling/SKILL.md`,
  `.cursor/skills/python-bindings/SKILL.md`,
  `.cursor/skills/lab-notebook/SKILL.md`,
  `.cursor/skills/weekly-checkpoint/SKILL.md`.

## 10. What I would do next

Two stretch directions, pick one paragraph: (a) drop to FP16 / BF16
inputs with FP32 accumulation and route through the tensor-core
WMMA API — this is the on-ramp to Week 5 (CUTLASS / CuTe / TMA on
Blackwell); (b) handle non-multiple-of-128 shapes via a tail-loop
epilogue without the if-branch killing throughput on the bulk
tiles. Either is enough; do **not** do both — the checkpoint is
about depth on one kernel, not breadth.

TODO
