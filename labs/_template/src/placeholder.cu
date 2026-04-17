// Placeholder. Replace with your kernel.
#include <cstddef>

__global__ void placeholder_kernel(float* y, const float* x, std::size_t n) {
    auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
