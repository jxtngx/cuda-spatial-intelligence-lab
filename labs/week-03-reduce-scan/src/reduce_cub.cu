// CUB baseline for the reduction perf target.
// Perf rule for the week: v4 must reach >= 80% of this kernel's throughput.
#include "reduce.hpp"

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

namespace week03 {

void launch_reduce_cub(const float* d_in, float* d_out, std::size_t n,
                       cudaStream_t stream) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n, stream);
    cudaMallocAsync(&d_temp, temp_bytes, stream);
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n, stream);
    cudaFreeAsync(d_temp, stream);
}

}  // namespace week03
