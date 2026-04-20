// Week 03 — parallel reduction API.
//
// Five stages, after Mark Harris's classic
// "Optimizing Parallel Reduction in CUDA" (still required reading).
// Each stage is a separate launcher so the bench/test can compare
// them apples-to-apples on the same input on the same stream.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace week03 {

enum class ReduceVersion : int {
    V0_InterleavedDivergent = 0,  // Harris stage 1: %-based stride, divergent warps
    V1_InterleavedStrided   = 1,  // Harris stage 2: strided index, no warp divergence
    V2_SequentialAddressing = 2,  // Harris stage 3: reverse stride, no bank conflicts
    V3_FirstAddDuringLoad   = 3,  // Harris stage 4: halve block count
    V4_WarpShuffle          = 4,  // Harris stage 7: __shfl_down_sync warp reduce + grid-stride
};

// Sum-reduce `n` floats from `d_in` into a single float at `*d_out`.
// Allocates no device memory (caller owns scratch via successive calls).
// Stream-ordered.
void launch_reduce(const float* d_in,
                   float* d_out,
                   std::size_t n,
                   ReduceVersion version,
                   cudaStream_t stream);

// Convenience: cub::DeviceReduce::Sum baseline. Implemented in reduce_cub.cu.
void launch_reduce_cub(const float* d_in,
                       float* d_out,
                       std::size_t n,
                       cudaStream_t stream);

}  // namespace week03
