// Lab 03 — inclusive scan API.
//
// Two implementations + a CUB baseline:
//   - hand-rolled Hillis-Steele block scan (PMPP 4e §11.2)
//   - cooperative_groups::inclusive_scan (CUDA C++ Programming Guide §B.18)
//
// For now both are single-block scans (n <= 1024). Multi-block decoupled
// look-back (Merrill-Garland) is a stretch goal in §10 What I would do next.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace week03 {

enum class ScanVersion : int {
    HillisSteele = 0,
    CoopGroups   = 1,
};

void launch_scan(const float* d_in,
                 float* d_out,
                 std::size_t n,        // n must be <= 1024 for the Tier-A scaffold
                 ScanVersion version,
                 cudaStream_t stream);

void launch_scan_cub(const float* d_in,
                     float* d_out,
                     std::size_t n,
                     cudaStream_t stream);

}  // namespace week03
