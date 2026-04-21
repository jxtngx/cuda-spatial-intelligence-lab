# Month 1 — C++20 + CUDA Foundations

Goal: by end of Month 1 you write modern C++20, you understand the GPU
memory hierarchy in your hands (not just in your head), and your tiled
single-precision GEMM hits ≥ 70% of cuBLAS on Spark.

> **Standing learning objective from Week 2 onward.** Every weekly lab
> from Week 2 to Week 16 must wrap its primary kernel as a PyTorch
> custom op via `torch.utils.cpp_extension`, with a `pytest` that
> verifies numerics from Python against a CPU reference. Performance
> requirement: Python wrapper overhead < 5% of kernel time at the
> largest test size. Pattern documented in
> `.cursor/skills/python-bindings/SKILL.md`. Week 1 is intentionally
> exempt — it stays pure C++/CUDA so the student can focus on RAII
> and the launch helper.

---

## Week 1 — Modern C++20 essentials + CUDA Hello

**Theme.** Get fluent in C++20 idioms you'll lean on for 16 weeks. Get
your first real CUDA kernel running on Spark with the right toolchain.

**Readings (do first).**
- Stroustrup, *PPP3* — Ch 4 (computation), Ch 5 (errors), Ch 9 (technicalities), Ch 12-14 (classes).
- Iglberger, *C++ Software Design* — Ch 1 (Software Design), Ch 2 (Build for Change), Ch 3 (Strategy / Command pattern).
- PMPP 4e — Ch 1 (Introduction), Ch 2 (Heterogeneous data parallel computing), Ch 3 (Multidimensional grids).
- CUDA C++ Programming Guide — §1, §2, §5 (programming model, programming interface, programming model details).

**Lab — `labs/lab-01-hello-cuda/`.**
1. Build a `DeviceBuffer<T>` RAII type around `cudaMallocAsync` /
   `cudaFreeAsync`. Move-only. Concept-constrained on `std::is_trivially_copyable`.
2. Build a `Stream` RAII type. Build a `KernelLaunch` helper that takes a
   designated-initializer config struct.
3. Implement vector add as the "hello, world" kernel using your wrappers.
4. Add a GoogleTest test with CPU reference + max-error assertion.

**Performance target.** Vector add at N=2^28 must achieve ≥ 85% of peak
DRAM bandwidth on Spark (Spark unified memory peak ≈ 273 GB/s).

**Deliverables.**
- `src/`, `bench/`, `tests/`, `report/LAB.md`.
- Nsight Systems trace + Nsight Compute report in `report/`.

**Rubric overlays.** Idiom axis specifically grades the wrappers — must
satisfy Rule of Five, must be `[[nodiscard]]` where appropriate, must
not leak on exception paths.

---

## Week 2 — Memory hierarchy and tiled GEMM

**Theme.** The single most important lesson in CUDA: shared memory and
tiling. By end of week, your naive GEMM has been replaced by a tiled
GEMM and you can articulate why each refactor moved the needle.

**Readings.**
- PMPP 4e — Ch 4 (Compute architecture and scheduling), Ch 5 (Memory architecture and data locality).
- CUDA C++ Best Practices Guide — §9 (Memory optimizations) end-to-end.
- Iglberger Ch 4-5 (Visitor / Strategy modernization).

**Lab — `labs/lab-02-tiled-gemm/`.**
1. Implement four versions of single-precision GEMM (M=N=K=4096):
   - `gemm_naive` (one thread = one output element, global memory).
   - `gemm_tiled_32` (32×32 tiles, shared memory).
   - `gemm_tiled_64` (64×64 tiles, padded for bank conflicts).
   - `gemm_tiled_async` (uses `cuda::memcpy_async` for global→shared).
2. Microbench all four against `cublasSgemm` baseline.
3. For each, commit a Nsight Compute report and a one-paragraph diagnosis.
4. **Python bindings (first time):** wrap `gemm_tiled_async` as a
   PyTorch custom op via `torch.utils.cpp_extension` (JIT `load()`
   acceptable for this week). Add `python/test_gemm.py` that calls
   the op on torch tensors and asserts max-abs-error against a CPU
   `torch.matmul` reference. See
   `.cursor/skills/python-bindings/SKILL.md`.

**Performance target.** `gemm_tiled_async` ≥ 50% of `cublasSgemm` on
M=N=K=4096. (Week 4 target: ≥ 70%.) Python wrapper overhead < 5% of
kernel time at M=N=K=4096.

---

## Week 3 — Reductions, scans, atomics, warp primitives

**Theme.** Most non-GEMM workloads are some flavor of reduce/scan/scatter.
Master the building blocks.

**Readings.**
- PMPP 4e — Ch 6 (Performance considerations), Ch 10 (Reduction), Ch 11 (Prefix sum / scan), Ch 9 (Parallel histogram, atomics).
- *Optimizing Parallel Reduction in CUDA* (Mark Harris classic; still required reading).
- CUDA C++ Programming Guide — §B.16 (warp shuffle), §B.18 (cooperative groups).

**Lab — `labs/lab-03-reduce-scan/`.**
1. Implement parallel reduction in 5 stages (Harris's classic), benchmark
   each, commit Nsight Compute reports.
2. Implement an inclusive scan using `cooperative_groups::scan` AND a
   hand-rolled Hillis-Steele variant. Compare to `cub::DeviceScan`.
3. Implement a histogram with warp aggregation.

**Performance target.** Final reduction ≥ 80% of `cub::DeviceReduce`.
Final scan ≥ 70% of `cub::DeviceScan`.

---

## Week 4 — Checkpoint: production-grade tiled GEMM

**Theme.** Take your Week 2 tiled GEMM all the way. This is the first
real test of whether you can drive a kernel to a perf target with
profile evidence.

**Readings (refresh).**
- PMPP 4e — Ch 5, 6 again. Yes, again.
- CUTLASS docs — read the "Efficient GEMM in CUDA" tutorial even though
  you won't use CUTLASS until Week 5; it teaches the vocabulary
  (warp tiles, MMA fragments, double buffering).

**Lab — `labs/lab-04-gemm-checkpoint/`.**
1. Take the best Week-2 GEMM and add: double-buffered async copies,
   register tiling (each thread computes a 4×4 sub-tile), vectorized
   loads (`float4`), and read-only `__ldg` where appropriate.
2. Sweep tile sizes (32, 64, 128) and report a small grid of
   GFLOPS/Watt-style numbers.
3. Compare against `cublasSgemm` AND a CUTLASS GEMM you build from a
   provided template.
4. Write a paper-style `report/LAB.md` with roofline analysis.

**Performance target (must hit).** ≥ 70% of `cublasSgemm` on
M=N=K=4096 single precision. Bonus: ≥ 80%.

**Checkpoint rubric.** Strict — needs **17/20**, not 14.
