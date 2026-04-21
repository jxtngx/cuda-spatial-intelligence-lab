// Week 03 — warp-aggregated histogram.
//
// 256-bin histogram of uint8_t inputs. Two versions:
//   - global atomics only (the "obvious" baseline)
//   - shared-memory privatized + warp aggregation (PMPP 4e §9)
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace week03 {

enum class HistVersion : int {
    GlobalAtomic = 0,
    SharedWarpAggregated = 1,
};

void launch_histogram(const std::uint8_t* d_in,
                      unsigned int* d_bins,  // length 256
                      std::size_t n,
                      HistVersion version,
                      cudaStream_t stream);

}  // namespace week03
