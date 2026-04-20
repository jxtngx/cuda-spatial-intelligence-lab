// 256-bin histogram, two versions.
//
// Tier A: both kernels compile and produce correct results. One
// TODO(student): marker below points at the warp-aggregation step that
// is the whole reason the privatized version exists.
#include "histogram.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace week03 {

namespace cg = cooperative_groups;

static constexpr int BINS = 256;
static constexpr int HBLOCK = 256;

__global__ void hist_global(const std::uint8_t* __restrict__ in,
                            unsigned int* __restrict__ bins, std::size_t n) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        atomicAdd(&bins[in[i]], 1u);
    }
}

__global__ void hist_shared_warp(const std::uint8_t* __restrict__ in,
                                 unsigned int* __restrict__ bins,
                                 std::size_t n) {
    __shared__ unsigned int smem_bins[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) smem_bins[b] = 0u;
    __syncthreads();

    // TODO(student): warp aggregation.
    // Naive form below uses one atomicAdd per input element on shared
    // memory. The optimized form (see PMPP 4e §9.5) has each warp
    // group threads writing the SAME bin via __ballot_sync /
    // __match_any_sync, elects one lane per group, and that lane does
    // a single atomicAdd of the group size. Implement that — it's
    // the difference between v4 and the perf target.
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        atomicAdd(&smem_bins[in[i]], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        if (smem_bins[b] != 0u) atomicAdd(&bins[b], smem_bins[b]);
    }
}

void launch_histogram(const std::uint8_t* d_in, unsigned int* d_bins,
                      std::size_t n, HistVersion version,
                      cudaStream_t stream) {
    cudaMemsetAsync(d_bins, 0, sizeof(unsigned int) * BINS, stream);
    if (n == 0) return;
    int grid = 1024;
    switch (version) {
        case HistVersion::GlobalAtomic:
            hist_global<<<grid, HBLOCK, 0, stream>>>(d_in, d_bins, n); break;
        case HistVersion::SharedWarpAggregated:
            hist_shared_warp<<<grid, HBLOCK, 0, stream>>>(d_in, d_bins, n); break;
    }
}

}  // namespace week03
