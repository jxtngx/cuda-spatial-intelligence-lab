# Week 01 — Modern C++20 essentials + CUDA Hello

> Source of truth: `.cursor/skills/curriculum-plan/month-1-foundations.md`
> (Week 1 section). This file is your local writeup template; treat it
> as a living lab notebook through the week and finalize with `/lab-report`.

## 1. Spec

Build a small, correct, idiomatic C++20 + CUDA foundation you'll reuse
all 16 weeks:

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
- `.cursor/skills/curriculum-plan/month-1-foundations.md` (Week 1).
- `.cursor/skills/cuda-kernel-authoring/SKILL.md` (the standard loop).
- `.cursor/skills/cpp20-modern-idioms/SKILL.md` (RAII wrappers).
