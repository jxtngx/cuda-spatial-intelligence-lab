// Five reduction kernels (Harris stages 1, 2, 3, 4, and 7).
//
// Tier A scaffold: every kernel compiles and the bench/test harness
// can already invoke them. Three TODO(student): markers below mark
// the inner-loop body, the launch config, and the warp-shuffle
// primitive choice. The first four kernels reduce a single block of
// `BLOCK` floats; the host launcher reduces in two passes for `n >
// BLOCK`. v4 uses a single-pass grid-stride loop and is what the perf
// target is measured against.
#include "reduce.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace week03 {

namespace cg = cooperative_groups;

static constexpr int BLOCK = 256;

// ---------- v0: interleaved addressing, divergent warps ----------
__global__ void reduce_v0(const float* __restrict__ in, float* __restrict__ out,
                          std::size_t n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // TODO(student): interleaved reduction with `if (tid % (2*s) == 0)`.
    // This is intentionally the SLOWEST version — branch-divergent
    // within a warp. PMPP 4e §10.2; Harris slide 8.
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v1: interleaved addressing, strided index ----------
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out,
                          std::size_t n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // Same arithmetic as v0 but the active threads are contiguous
    // (`index < blockDim.x`), eliminating warp divergence.
    // Harris slide 14; PMPP 4e §10.3.
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        unsigned index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v2: sequential addressing (no bank conflicts) ----------
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out,
                          std::size_t n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // TODO(student): reverse-stride loop `for (s = blockDim.x/2; s > 0; s >>= 1)`
    // with `if (tid < s) sdata[tid] += sdata[tid + s];`. This eliminates
    // shared-memory bank conflicts. Harris slide 22; PMPP 4e §10.4.
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v3: first add during load (halve block count) ----------
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out,
                          std::size_t n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    // Each block now covers 2*BLOCK input elements; one add at load time.
    unsigned i = blockIdx.x * (blockDim.x * 2) + tid;
    float a = (i < n) ? in[i] : 0.0f;
    float b = (i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v4: warp shuffle + grid-stride (final stage) ----------
__device__ inline float warp_reduce_sum(float v) {
    // TODO(student): replace this loop with three __shfl_down_sync calls
    // for widths 16, 8, 4, 2, 1 over the FULL 0xFFFFFFFF mask. The loop
    // form below is correct and compiles; the explicit unroll is what
    // the rubric grades on the Idiom axis. CUDA C++ Programming Guide
    // §B.16; Harris slide 35.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out,
                          std::size_t n) {
    __shared__ float warp_sums[BLOCK / 32];
    unsigned tid = threadIdx.x;
    unsigned lane = tid & 31u;
    unsigned warp = tid >> 5;

    float sum = 0.0f;
    // Grid-stride loop: each thread accumulates many input elements.
    for (std::size_t i = blockIdx.x * blockDim.x + tid;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        sum += in[i];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (tid < BLOCK / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) atomicAdd(out, sum);
    }
}

// ---------- host launcher ----------

static int compute_grid_v01234(std::size_t n, ReduceVersion v) {
    switch (v) {
        case ReduceVersion::V3_FirstAddDuringLoad:
            return static_cast<int>((n + 2 * BLOCK - 1) / (2 * BLOCK));
        case ReduceVersion::V4_WarpShuffle:
            // Cap grid so each thread sees ~enough work; tuned for sm_121.
            return 1024;
        default:
            return static_cast<int>((n + BLOCK - 1) / BLOCK);
    }
}

void launch_reduce(const float* d_in, float* d_out, std::size_t n,
                   ReduceVersion version, cudaStream_t stream) {
    if (n == 0) {
        cudaMemsetAsync(d_out, 0, sizeof(float), stream);
        return;
    }

    int grid = compute_grid_v01234(n, version);

    if (version == ReduceVersion::V4_WarpShuffle) {
        // Single-pass: zero the accumulator then atomicAdd into it.
        cudaMemsetAsync(d_out, 0, sizeof(float), stream);
        // TODO(student): tune the launch config. Try grid=1024, BLOCK=256;
        // sweep grid in {256, 512, 1024, 2048} and report which one wins
        // in §7 Results. PMPP 4e §6.3 (occupancy); Nsight Compute
        // "Launch Statistics" section.
        reduce_v4<<<grid, BLOCK, 0, stream>>>(d_in, d_out, n);
        return;
    }

    // Two-pass for v0..v3: per-block partials, then a final v4 reduction
    // of those partials so we don't need a host round-trip.
    float* d_partials = nullptr;
    cudaMallocAsync(&d_partials, sizeof(float) * grid, stream);

    switch (version) {
        case ReduceVersion::V0_InterleavedDivergent:
            reduce_v0<<<grid, BLOCK, 0, stream>>>(d_in, d_partials, n); break;
        case ReduceVersion::V1_InterleavedStrided:
            reduce_v1<<<grid, BLOCK, 0, stream>>>(d_in, d_partials, n); break;
        case ReduceVersion::V2_SequentialAddressing:
            reduce_v2<<<grid, BLOCK, 0, stream>>>(d_in, d_partials, n); break;
        case ReduceVersion::V3_FirstAddDuringLoad:
            reduce_v3<<<grid, BLOCK, 0, stream>>>(d_in, d_partials, n); break;
        default: break;
    }

    cudaMemsetAsync(d_out, 0, sizeof(float), stream);
    int grid2 = 1;
    int block2 = (grid >= 1024) ? 1024 : ((grid + 31) / 32) * 32;
    if (block2 < 32) block2 = 32;
    // Reuse v4 to fold the partials.
    reduce_v4<<<grid2, BLOCK, 0, stream>>>(d_partials, d_out,
                                           static_cast<std::size_t>(grid));
    cudaFreeAsync(d_partials, stream);
}

}  // namespace week03
