# Lab 01 — Glossary

> Only **new terms** introduced in this lab.
> Future labs will not redefine these.
> Each entry: one-line definition, where it first appears in this lab, and a primary-source citation.
>
> Read this before [`LAB.md`](./LAB.md) §1 Spec.

## CUDA

- **kernel** — a function annotated `__global__` that runs on the GPU, invoked from host code with `<<<grid, block, shmem, stream>>>` launch syntax.
  First appears in: `src/axpy.cu`.
  See: PMPP 4e §2.4; CUDA C++ Programming Guide §2.1.
- **kernel launch** — the host-side `kernel<<<...>>>(args)` call that enqueues `grid * block` threads onto a stream, asynchronous w.r.t. the host.
  First appears in: `src/axpy.cu` (via `launch<>` helper).
  See: CUDA C++ Programming Guide §2.2 + §6.2.3.
- **stream** — a FIFO of GPU work that executes in order on the device; operations across different streams may overlap.
  First appears in: `src/stream.hpp`.
  See: CUDA C++ Programming Guide §3.2.6.
- **`cudaMallocAsync`** — stream-ordered device allocation backed by a memory pool; pairs with `cudaFreeAsync` and is cheaper than `cudaMalloc` for repeated allocations.
  First appears in: `src/device_buffer.hpp`.
  See: CUDA C++ Programming Guide §11 (memory pools).
- **`cudaFreeAsync`** — stream-ordered free that returns memory to the pool when prior work on the stream completes.
  First appears in: `src/device_buffer.hpp`.
  See: CUDA C++ Programming Guide §11.
- **grid-stride loop** — kernel pattern where each thread processes multiple elements `i, i+stride, i+2*stride, ...` with `stride = gridDim.x * blockDim.x`, decoupling launch shape from `N`.
  First appears in: `src/axpy.cu` v2.
  See: PMPP 4e §3.5; NVIDIA blog "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops".
- **coalesced access** — memory access pattern where the 32 threads of a warp issue contiguous, aligned global-memory loads/stores that collapse to the minimum number of DRAM transactions.
  First appears in: `src/axpy.cu` v0 hypothesis.
  See: PMPP 4e §5.3.
- **`float4` vectorized load** — a single 16-byte load that fetches four `float`s, halving instruction count and improving memory efficiency on bandwidth-bound kernels.
  First appears in: `src/axpy.cu` v1.
  See: PMPP 4e §5.4; CUDA C++ Programming Guide §6.2.4.
- **`__align__(16)`** — host-side / type alignment attribute that guarantees a 16-byte boundary, prerequisite for safe `float4` loads.
  First appears in: `src/device_buffer.hpp`.
  See: CUDA C++ Programming Guide §6.2.4.
- **occupancy** — ratio of active warps per SM to the theoretical max, bounded by registers, shared memory, and block size; higher is often (not always) better for memory-bound kernels.
  First appears in: `LAB.md` §3 Method.
  See: PMPP 4e §4.6; Nsight Compute "Occupancy" section.
- **achieved bandwidth (GB/s)** — measured DRAM throughput, `(bytes_moved) / (time_seconds) / 1e9`; the denominator of the Lab 01 perf target.
  First appears in: `bench/bench_axpy.cpp`.
  See: Nsight Compute "Memory Workload Analysis".
- **Speed of Light (SOL)** — Nsight Compute section that reports the kernel's achieved fraction of theoretical compute and memory peaks; for AXPY it should saturate the memory SOL, not compute.
  First appears in: `LAB.md` §2 Hypothesis.
  See: Nsight Compute User Guide, "Speed of Light" section.
- **Memory Workload Analysis** — Nsight Compute section that breaks down DRAM/L2/L1 traffic, hit rates, and sector utilization; the evidence required for the Lab 01 perf claim.
  First appears in: `LAB.md` §3 Spec.
  See: Nsight Compute User Guide.

## C++20

- **RAII (Resource Acquisition Is Initialization)** — idiom where a resource's lifetime is tied to an object's lifetime; the destructor releases the resource.
  Applied here to `cudaStream_t` and device memory.
  First appears in: `src/device_buffer.hpp`, `src/stream.hpp`.
  See: Stroustrup PPP3 Ch 13; Iglberger Ch 1; cppreference ["RAII"](https://en.cppreference.com/w/cpp/language/raii).
- **move-only type** — a type whose copy constructor and copy assignment are deleted but move operations are defined; used to express unique ownership of a resource.
  First appears in: `src/device_buffer.hpp`.
  See: cppreference ["Move constructors"](https://en.cppreference.com/w/cpp/language/move_constructor).
- **`std::span<T>`** — a non-owning view over a contiguous sequence of `T` with size known at runtime; C++20 standard library type, used here as the kernel-argument view into device buffers.
  First appears in: `src/axpy.hpp`.
  See: [cppreference `std::span`](https://en.cppreference.com/w/cpp/container/span).
- **concept** — a named compile-time predicate over template parameters, enforced via `requires` clauses or `template <Concept T>`; used here to constrain `DeviceBuffer<T>` on `std::is_trivially_copyable_v<T>`.
  First appears in: `src/device_buffer.hpp`.
  See: [cppreference "Constraints and concepts"](https://en.cppreference.com/w/cpp/language/constraints).
- **`std::is_trivially_copyable_v<T>`** — type-trait variable template that is `true` if `T` can be copied byte-for-byte (a precondition for `cudaMemcpy`).
  First appears in: `src/device_buffer.hpp`.
  See: [cppreference `std::is_trivially_copyable`](https://en.cppreference.com/w/cpp/types/is_trivially_copyable).
- **designated initializers** — C++20 syntax `T{.field = value}` that initializes specific members by name; used here for the `LaunchConfig` struct passed to `launch<>`.
  First appears in: `src/axpy.cu` (call site of `launch<>`).
  See: [cppreference "Aggregate initialization"](https://en.cppreference.com/w/cpp/language/aggregate_initialization).
- **templated kernel** — a `__global__` function template specialized on element type (`float`, `__half`, `__nv_bfloat16` here) so one source produces multiple instantiations.
  First appears in: `src/axpy.cu`.
  See: PMPP 4e §3.6; CUDA C++ Programming Guide §14.5.

## Spatial Intelligence / CV

*No new terms introduced this week.*
*Lab 01 is a pure C++20 + CUDA foundations lab; spatial-intelligence and CV vocabulary starts in Month 3.*
