---
name: cuda-kernel-authoring
description: Step-by-step workflow for writing a new CUDA kernel from spec to profile-validated implementation on DGX Spark (sm_121). Use when starting a new kernel, refactoring an existing one, or porting Python/PyTorch logic to CUDA.
---

# Authoring a CUDA kernel (Spark / sm_121)

This is the canonical loop. Do not skip steps. Do not write the kernel before
writing the test.

## 0. Preconditions

- You're in a `labs/week-NN-*/` folder with a `CMakeLists.txt` already
  targeting `CMAKE_CUDA_ARCHITECTURES "121"` and `CMAKE_CXX_STANDARD 20`.
- Nsight Systems and Nsight Compute are on `$PATH` (`nsys`, `ncu`).

## 1. Write the spec

Before any code, write into `report/LAB.md`:

```
### Kernel spec: <name>
- Inputs: shape(s), dtype(s), layout (row/col-major or arbitrary)
- Output: shape, dtype, layout
- Numerical tolerance (max abs error vs CPU reference)
- Performance target (e.g. ≥ 70% of cuBLAS sgemm; or ≥ 80% of peak DRAM BW)
- Constraints (in-place? fixed sizes? streamable?)
```

If you can't fill that out, you can't write the kernel.

## 2. Write the CPU reference + test first

`tests/test_<name>.cpp` (GoogleTest):

```cpp
TEST(KernelName, MatchesReference) {
    auto x = make_random<float>(N, /*seed=*/42);
    auto y_cpu = cpu_reference(x);
    auto y_gpu = run_kernel(x);
    EXPECT_LT(max_abs_diff(y_cpu, y_gpu), kTolerance);
}
```

Test at least: power-of-two N, prime N, N=1, N=large (≥ 2^24).

## 3. Write the kernel — naive first

Get correctness without optimizing. Use `cudaMallocManaged` for the
prototype (Spark unified memory is fast). Add `compute-sanitizer`:

```bash
compute-sanitizer --tool memcheck   ./tests/test_<name>
compute-sanitizer --tool racecheck  ./tests/test_<name>
```

If sanitizers fire, fix before benching.

## 4. Microbench

`bench/bench_<name>.cpp` using `cudaEvent_t` timing or `nvbench`. Report
GFLOP/s or GB/s vs roofline. Add NVTX ranges:

```cpp
nvtxRangePushA("kernel_<name>");
launch(kernel, ...)(args...);
cudaStreamSynchronize(s);
nvtxRangePop();
```

## 5. Optimize in named passes — one per commit

Pass list (apply only those that match your kernel category):

1. **Coalesce** global memory accesses (PMPP §5.3).
2. **Tile** with shared memory (PMPP §5.4).
3. **Pad** to kill bank conflicts (PMPP §5.5).
4. **Vectorize** loads (`float4`, `__align__`).
5. **Async-copy** global → shared (`cuda::memcpy_async` /
   `cp.async.bulk` for TMA).
6. **Double-buffer** so compute overlaps with the next async copy.
7. **Register-tile**: each thread computes a small sub-tile.
8. **Cooperative groups** for warp-level reductions.
9. **Tensor cores** (WMMA / `mma.sync` / CUTLASS) for any matmul shape.
10. **TMA + thread-block clusters** (Blackwell) when you've earned it.

After each pass: `compute-sanitizer`, then `ncu` report into `report/`.

## 6. Profile and decide

Run the standard Nsight Compute set:

```bash
ncu --set full --section SpeedOfLight --section Occupancy \
    --section MemoryWorkloadAnalysis --section WarpStateStats \
    --import-source on -k <name> -c 5 \
    -o report/ncu_<name>_v<N>.ncu-rep ./bench/bench_<name>
```

Read in this order: Speed of Light → Roofline → Occupancy → Warp State
Stats → Memory Workload → Source counters.

Hand off to the `cuda-perf-profiler` subagent if you can't name the
bottleneck in one sentence.

## 7. Stop at the target

When you hit the lab's perf target, stop. Write the result into
`report/LAB.md`. Move on. **Do not optimize past the target** — the
curriculum has more for you to learn.

## C++20 wrapper standards (use these, don't reinvent)

```cpp
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n, cudaStream_t s = nullptr) : n_{n} {
        CUDA_CHECK(cudaMallocAsync(&p_, n * sizeof(T), s));
    }
    ~DeviceBuffer() { if (p_) cudaFreeAsync(p_, nullptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_{std::exchange(o.p_, nullptr)}, n_{o.n_} {}
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    T* data() noexcept { return p_; }
    std::size_t size() const noexcept { return n_; }
    std::span<T> span() noexcept { return {p_, n_}; }
private:
    T* p_{nullptr};
    std::size_t n_{0};
};
```

## Hand-offs

- "I can't name the bottleneck" → `cuda-perf-profiler`.
- "Is this idiomatic C++?" → `cpp20-tutor`.
- "Did I miss a CUDA pattern?" → `cuda-tutor`.
- "Is this ready to merge?" → `cuda-code-reviewer`.
