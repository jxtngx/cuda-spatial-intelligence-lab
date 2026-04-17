---
name: cpp20-modern-idioms
description: Reference of modern C++20 idioms used throughout the lab — RAII wrappers for CUDA resources, concept-constrained templates, ranges-based pipelines, std::span/mdspan views, designated-initializer config structs, and std::expected-style error handling. Use when writing or reviewing host-side C++.
---

# C++20 idioms for this lab

The host-side C++ in this repo is opinionated. These idioms appear in every
lab. Use them; don't reinvent them.

## RAII wrappers for CUDA resources

Every CUDA handle gets an RAII owner. Move-only. Rule of Five.

```cpp
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
    ~Stream() { if (s_) cudaStreamDestroy(s_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& o) noexcept : s_{std::exchange(o.s_, nullptr)} {}
    Stream& operator=(Stream&& o) noexcept {
        if (this != &o) { if (s_) cudaStreamDestroy(s_); s_ = std::exchange(o.s_, nullptr); }
        return *this;
    }
    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }
    void sync() { CUDA_CHECK(cudaStreamSynchronize(s_)); }
private:
    cudaStream_t s_{nullptr};
};
```

Same shape for `Event`, `Graph`, `GraphExec`, `MemoryPool`, TRT
`IRuntime`/`IExecutionContext`, etc.

## Concepts on templates (no SFINAE)

```cpp
template <typename T>
concept GpuScalar = std::is_arithmetic_v<T>
                 && std::is_trivially_copyable_v<T>
                 && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);

template <GpuScalar T>
void axpy(T alpha, std::span<const T> x, std::span<T> y, Stream& s);
```

For a kernel that only makes sense for floating-point:

```cpp
template <std::floating_point T> requires (sizeof(T) <= 4)
void gemm(...);
```

## `std::span` (and `std::mdspan` when available)

Pass non-owning views into kernels' host launchers. Don't pass raw `T*` +
`size_t` pairs.

```cpp
void axpy(float alpha, std::span<const float> x, std::span<float> y, Stream& s);
```

`std::mdspan` (C++23, but available as `std::experimental::mdspan` /
`Kokkos::mdspan` polyfill) is excellent for matrix shapes:

```cpp
using MatrixView = std::mdspan<float, std::dextents<int, 2>>;
void gemm(MatrixView A, MatrixView B, MatrixView C, Stream& s);
```

## Designated initializers for launch configs

Kill the `<<<grid, block, smem, stream>>>` magic-quadruple-positional pain:

```cpp
struct Launch {
    dim3 grid{1};
    dim3 block{256};
    std::size_t shared_bytes{0};
    cudaStream_t stream{nullptr};
};

template <auto Kernel, typename... Args>
void launch(Launch cfg, Args&&... args) {
    Kernel<<<cfg.grid, cfg.block, cfg.shared_bytes, cfg.stream>>>(std::forward<Args>(args)...);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Usage:
launch<gemm_kernel>({.grid={M/32, N/32}, .block=256, .stream=s.get()}, A, B, C);
```

## `std::expected`-style error handling on the host

CUDA APIs return `cudaError_t`. Wrap them in a typed-error monadic style:

```cpp
template <typename T>
using Result = std::expected<T, std::string>;   // C++23; polyfill available

inline Result<void> cuda_check(cudaError_t e, const char* what) {
    return e == cudaSuccess ? Result<void>{} : std::unexpected(std::string{what} + ": " + cudaGetErrorString(e));
}
```

Reserve exceptions for genuinely exceptional conditions (allocation
failure during construction). Use `expected` in code paths where errors
are control flow.

## Strong types over raw `int`

```cpp
struct DeviceId { int v; explicit operator int() const noexcept { return v; } };
struct StreamId { int v; explicit operator int() const noexcept { return v; } };
```

Stops you passing a stream id where a device id was expected.

## Free functions over methods

`Tensor::gemm(Tensor&)` is wrong. `Tensor` owns memory; `gemm(...)` is a
free function that takes views. (Iglberger Guideline 15: design for the
addition of operations.)

## Ranges for host-side data prep

```cpp
auto valid_indices = std::views::iota(0, N)
                   | std::views::filter([&](int i){ return mask[i]; })
                   | std::views::transform([&](int i){ return remap[i]; });
```

You'll want this for SfM correspondence prep, RANSAC inlier filtering,
etc.

## Avoid

- `using namespace std;` in headers.
- Raw `new`/`delete` outside of placement-new in container internals.
- `int` for sizes that can exceed 2^31 — use `std::ptrdiff_t` or `std::size_t`.
- Inheritance for code reuse — use composition.
- Header-only "for convenience" classes that allocate on the GPU. Keep
  GPU resources owned by named, RAII-managed types in `.cpp`/`.cu`.
