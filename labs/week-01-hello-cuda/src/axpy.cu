#include "axpy.hpp"
#include "cuda_check.hpp"

#include <cstddef>

namespace cudalab {

namespace {

template <typename T>
__global__ void axpy_v0_naive(T alpha, const T* __restrict__ x,
                              T* __restrict__ y, std::size_t n) {
    auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

// Vectorized via float4 (T==float). 4 elements per thread.
__global__ void axpy_v1_vec4_f32(float alpha, const float* __restrict__ x,
                                 float* __restrict__ y, std::size_t n4) {
    auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 xv = reinterpret_cast<const float4*>(x)[i];
        float4 yv = reinterpret_cast<float4*>(y)[i];
        yv.x = alpha * xv.x + yv.x;
        yv.y = alpha * xv.y + yv.y;
        yv.z = alpha * xv.z + yv.z;
        yv.w = alpha * xv.w + yv.w;
        reinterpret_cast<float4*>(y)[i] = yv;
    }
}

__global__ void axpy_v2_stride_f32(float alpha, const float* __restrict__ x,
                                   float* __restrict__ y, std::size_t n4) {
    std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         i < n4; i += stride) {
        float4 xv = reinterpret_cast<const float4*>(x)[i];
        float4 yv = reinterpret_cast<float4*>(y)[i];
        yv.x = alpha * xv.x + yv.x;
        yv.y = alpha * xv.y + yv.y;
        yv.z = alpha * xv.z + yv.z;
        yv.w = alpha * xv.w + yv.w;
        reinterpret_cast<float4*>(y)[i] = yv;
    }
}

constexpr unsigned kBlock = 256;

}  // namespace

template <>
void axpy<float>(float alpha, std::span<const float> x, std::span<float> y,
                 Stream& s, AxpyVersion v) {
    auto n = y.size();
    if (n == 0) return;

    switch (v) {
        case AxpyVersion::v0_naive: {
            unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);
            axpy_v0_naive<float>
                <<<grid, kBlock, 0, s.get()>>>(alpha, x.data(), y.data(), n);
            break;
        }
        case AxpyVersion::v1_vec4: {
            // Falls back to naive if not 4-aligned.
            if ((n % 4) != 0) {
                unsigned grid =
                    static_cast<unsigned>((n + kBlock - 1) / kBlock);
                axpy_v0_naive<float><<<grid, kBlock, 0, s.get()>>>(
                    alpha, x.data(), y.data(), n);
            } else {
                std::size_t n4 = n / 4;
                unsigned grid =
                    static_cast<unsigned>((n4 + kBlock - 1) / kBlock);
                axpy_v1_vec4_f32<<<grid, kBlock, 0, s.get()>>>(
                    alpha, x.data(), y.data(), n4);
            }
            break;
        }
        case AxpyVersion::v2_stride: {
            if ((n % 4) != 0) {
                unsigned grid =
                    static_cast<unsigned>((n + kBlock - 1) / kBlock);
                axpy_v0_naive<float><<<grid, kBlock, 0, s.get()>>>(
                    alpha, x.data(), y.data(), n);
            } else {
                std::size_t n4 = n / 4;
                // Grid sized for ~16 waves on a Blackwell SM count;
                // tune by sweep.
                unsigned grid =
                    static_cast<unsigned>(std::min<std::size_t>(
                        (n4 + kBlock - 1) / kBlock, 65535));
                axpy_v2_stride_f32<<<grid, kBlock, 0, s.get()>>>(
                    alpha, x.data(), y.data(), n4);
            }
            break;
        }
    }
    check_last_error("axpy launch");
}

}  // namespace cudalab
