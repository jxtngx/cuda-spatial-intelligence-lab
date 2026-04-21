# Lab 01 — Modern C++20 essentials + CUDA Hello

> Source of truth: `.cursor/skills/curriculum-plan/month-1-foundations.md`
> (Lab 1 section). This file is your local writeup template; treat it
> as a living lab notebook through the lab and finalize with `/lab-report`.

## 0. Tasks — what you actually do in this lab

The `src/`, `tests/`, and `bench/` trees ship with a working reference
implementation of `axpy<float>` (v0 naive, v1 vec4, v2 grid-stride),
the `Stream` / `DeviceBuffer<T>` RAII wrappers, and a GoogleTest +
microbench. Your job in this lab is to **bring it up on your Spark,
prove it hits the perf target with profile evidence, close the gaps
between scaffold and spec, and write the result up**.

Work the list top-to-bottom. Don't move past a checkbox until it's
green.

### A. Read first (do not skip)

- [ ] Stroustrup *PPP3* Ch 4, 5, 9, 12-14.
- [ ] Iglberger *C++ Software Design* Ch 1-3.
- [ ] PMPP 4e Ch 1, 2, 3, and §5.3 (coalesced access).
- [ ] CUDA C++ Programming Guide §1, §2, §3.2.6 (streams), §11
      (memory pools / `cudaMallocAsync`).
- [ ] Skim `.cursor/skills/cpp20-modern-idioms/SKILL.md` and
      `.cursor/skills/cuda-kernel-authoring/SKILL.md`.

### B. Bring the scaffold up on Spark

- [ ] Confirm Spark toolchain per `.cursor/skills/dgx-spark-setup/SKILL.md`
      (CUDA 13, CMake ≥ 3.28, Ninja, Nsight host tools, sm_121).
- [ ] From `labs/lab-01-hello-cuda/`, configure & build:
      `cmake -S . -B build -G Ninja && cmake --build build -j`.
- [ ] Run the tests: `ctest --test-dir build --output-on-failure`.
      All `AxpyAtSize/{Naive,Vec4,Stride}` cases must pass at sizes
      {1, 17, 2^16, 2^28}, plus `AxpyEdge.ZeroSizeIsNoop`.
- [ ] Read every file in `src/` and write a one-sentence comment in
      your own words at the top of `axpy.cu`, `device_buffer.hpp`,
      and `stream.hpp` describing what it does and *why it is shaped
      that way*. (You will reuse these wrappers for 15 more labs.)

### C. Close the scaffold ↔ spec gaps

The reference implementation under-delivers vs. §1 Spec in three
places. Fix them before claiming the lab is done.

- [ ] **`launch<Kernel>(LaunchConfig{}, args...)` helper.** §1 and
      `GLOSSARY.md` describe a designated-initializer launch helper;
      it does not exist. Add it (suggested home: `src/launch.hpp`)
      and refactor `axpy.cu` to call it instead of raw `<<<...>>>`.
- [ ] **`__half` and `__nv_bfloat16` instantiations.** §1 says
      `T ∈ {float, __half, __nv_bfloat16}`. Add the two missing
      explicit instantiations in `axpy.cu` and matching test cases
      with the looser tolerance (≤ 1e-2) in §1.
- [ ] **Block-size sweep.** §3 v2 says "sweep {128, 256, 512}".
      `kBlock = 256` is currently hardcoded. Either templatize on
      block size or add a runtime parameter so the bench can sweep
      and the §4 table can be filled honestly.

### D. Run the bench and hit the perf target

- [ ] Run `./build/bench_axpy` and capture v0/v1/v2 numbers at
      N = 2^28. Paste them into the §4 table.
- [ ] Sweep block size {128, 256, 512} for v2 and pick the winner.
- [ ] Confirm v2 ≥ **232 GB/s** (≥ 85% of the ~273 GB/s Spark peak).
      If it doesn't, iterate (vector width, grid sizing, occupancy)
      until it does — and document each attempt in §5.

### E. Profile (evidence for the perf claim)

Follow `.cursor/skills/nsight-profiling/SKILL.md`. Commit the raw
reports under `report/` (create the directory).

- [ ] `report/nsys_axpy.qdrep` — Nsight Systems trace of the bench
      run, showing kernel timeline + memcopies.
- [ ] `report/ncu_axpy_v2.ncu-rep` — Nsight Compute report on v2 at
      N = 2^28 with at least the *Speed of Light* and *Memory
      Workload Analysis* sections populated.
- [ ] In §5, quote the achieved DRAM throughput from the
      `Memory Workload Analysis` section as your evidence of hitting
      the target.

### F. Write it up

- [ ] Fill in §2 Hypothesis *before* you look at the §4 numbers (if
      you peeked already, write what you would have predicted and be
      honest about it in §5).
- [ ] Fill in §4 Results table from the bench output.
- [ ] Fill in §5 Discussion answering: did SOL saturate memory?
      where did vectorization actually help — fewer instructions,
      fewer transactions, or both? did L2 hit rate matter?
- [ ] Fill in §6 What I would do next.
- [ ] Run `/lab-report` to polish the final writeup
      (`.cursor/skills/lab-notebook/SKILL.md`).

### G. Self-grade

- [ ] Run `/checkpoint` against the 5-axis rubric in
      `.cursor/skills/weekly-checkpoint/SKILL.md`. You need ≥ 14/20
      to advance to Lab 02 (≥ 17/20 in checkpoint labs 4/8/12/16).

### Definition of Done for Lab 01

All boxes A-G checked, `LAB.md` §2-§6 fully written, `report/`
contains both Nsight artifacts, `bench_axpy` shows v2 ≥ 232 GB/s
on this Spark, and `ctest` is green.

---

## 1. Spec

Build a small, correct, idiomatic C++20 + CUDA foundation you'll reuse
across all 16 labs:

- `DeviceBuffer<T>` — RAII owner around `cudaMallocAsync` / `cudaFreeAsync`.
  Move-only. Concept-constrained on `std::is_trivially_copyable_v<T>`.
- `Stream` — RAII owner around `cudaStream_t`. Move-only.
- `launch<Kernel>(LaunchConfig{}, args...)` — kernel-launch helper that
  takes a designated-initializer config struct.
- A first real kernel: `axpy` (`y = alpha * x + y`), templated on the
  element type, `std::span` views into device buffers.
- A GoogleTest unit test with a CPU reference and a max-error assertion
  at sizes {1, 17 (prime), 2^16, 2^28}.
- A microbenchmark that times `axpy` and reports achieved GB/s.

### Inputs / outputs / dtypes

`axpy<T>(T alpha, std::span<const T> x, std::span<T> y, Stream& s)`,
`T ∈ {float, __half, __nv_bfloat16}`, all sizes contiguous.

### Numerical tolerance

Max abs error vs CPU reference: ≤ 1e-5 for `float`; ≤ 1e-2 for `__half`
and `__nv_bfloat16`.

### Performance target

Achieve **≥ 85% of peak DRAM bandwidth on Spark unified memory**
(approx ≥ 232 GB/s of an estimated ~273 GB/s peak) on `axpy<float>` at
N = 2^28. Confirm with the microbench's reported GB/s and a Nsight
Compute Memory Workload Analysis section.

## 2. Hypothesis

*Write before optimizing.* AXPY is bandwidth-bound by construction
(2 reads + 1 write, ~2 FLOPs per element). Dominant cost is global
memory traffic; shared memory does not help. Achieving 85%+ of peak BW
should require coalesced loads, vectorized loads (`float4`), and a
block size that gives high occupancy. Nsight Compute should report
`Memory Workload Analysis` near peak DRAM throughput and `Speed of
Light` saturated on memory.

## 3. Method

### v0 — naive

One thread per element, scalar load/store. Establish correctness
baseline.

### v1 — vectorized

Use `float4` loads/stores when N is a multiple of 4 and pointers are
16-byte aligned (use `__align__(16)` allocations). Each thread now
handles 4 elements.

### v2 — grid-stride loop

Grid-stride loop so the launch is decoupled from N. Pick block size by
sweeping {128, 256, 512} and reporting in the table.

## 4. Results

Fill in after running the bench:

| Version | Time (ms, N=2^28) | Achieved BW (GB/s) | % of peak (~273 GB/s) |
|---|---|---|---|
| v0 naive       |  |  |  |
| v1 vectorized  |  |  |  |
| v2 grid-stride |  |  |  |

Reference Nsight reports:
- `report/nsys_axpy.qdrep`
- `report/ncu_axpy_v2.ncu-rep`

## 5. Discussion

Did the predicted bottleneck show up in `Speed of Light`? Where did
vectorization actually help — fewer instructions, fewer memory
transactions, or both? Did the L2 hit rate matter at all (it shouldn't
for a streaming workload — verify)?

## 6. What I would do next

Extend `axpy` to a 4-stream pipeline that overlaps `H→D` (or page
migration on Spark) with compute on chunks of N. Predict the speedup
ceiling first, then measure.

## 7. References

- PMPP 4e — Ch 2 (Heterogeneous data parallel computing), Ch 3
  (Multidimensional grids), Ch 5 §5.3 (coalesced access).
- Iglberger — Ch 1 ("The Art of Software Design"), Ch 3 (the Strategy
  pattern → applied here as a configurable launch helper).
- Stroustrup PPP3 — Ch 12-14 (classes), Ch 19 (vectors and free store).
- CUDA C++ Programming Guide — §3.2.6 (streams), §11 (memory pools /
  `cudaMallocAsync`).
- `.cursor/skills/curriculum-plan/month-1-foundations.md` (Lab 01 / Week 1).
- `.cursor/skills/cuda-kernel-authoring/SKILL.md` (the standard loop).
- `.cursor/skills/cpp20-modern-idioms/SKILL.md` (RAII wrappers).
