// CUB baseline for the scan perf target.
// v_cg / v_hs are graded against this at >= 70% throughput.
#include "scan.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace week03 {

void launch_scan_cub(const float* d_in, float* d_out, std::size_t n,
                     cudaStream_t stream) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n, stream);
    cudaMallocAsync(&d_temp, temp_bytes, stream);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n, stream);
    cudaFreeAsync(d_temp, stream);
}

}  // namespace week03
