#include "axpy.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "stream.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

using cudalab::axpy;
using cudalab::AxpyVersion;
using cudalab::DeviceBuffer;
using cudalab::Stream;

namespace {

float bench_one(AxpyVersion v, std::size_t n, int trials = 50) {
    Stream s;
    DeviceBuffer<float> d_x(n, s);
    DeviceBuffer<float> d_y(n, s);

    std::vector<float> h(n, 1.0f);
    d_x.copy_from_host(h, s);
    d_y.copy_from_host(h, s);
    s.sync();

    cudaEvent_t e0{}, e1{};
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Warmup.
    for (int i = 0; i < 5; ++i)
        axpy<float>(0.5f, d_x.span(), d_y.span(), s, v);
    s.sync();

    CUDA_CHECK(cudaEventRecord(e0, s.get()));
    for (int i = 0; i < trials; ++i)
        axpy<float>(0.5f, d_x.span(), d_y.span(), s, v);
    CUDA_CHECK(cudaEventRecord(e1, s.get()));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    return ms / static_cast<float>(trials);
}

const char* name(AxpyVersion v) {
    switch (v) {
        case AxpyVersion::v0_naive:  return "v0_naive";
        case AxpyVersion::v1_vec4:   return "v1_vec4";
        case AxpyVersion::v2_stride: return "v2_stride";
    }
    return "?";
}

}  // namespace

int main() {
    constexpr std::size_t N = std::size_t{1} << 28;  // 256M floats = 1 GiB
    // bytes moved per call: 2 reads + 1 write of float = 12 * N bytes
    const double bytes_per_call = 12.0 * static_cast<double>(N);

    std::printf("axpy<float>  N=%zu  (1.0 GiB y, 1.0 GiB x)\n", N);
    std::printf("Spark unified-memory peak BW ≈ 273 GB/s; target ≥ 232 GB/s "
                "(85%% of peak).\n");
    std::printf("%-12s | %12s | %12s | %12s\n", "version", "ms/call",
                "GB/s", "% of peak");

    for (auto v : {AxpyVersion::v0_naive, AxpyVersion::v1_vec4,
                   AxpyVersion::v2_stride}) {
        float ms = bench_one(v, N);
        double gbps = bytes_per_call / (ms * 1e-3) / 1e9;
        double pct  = 100.0 * gbps / 273.0;
        std::printf("%-12s | %12.3f | %12.2f | %11.1f%%\n", name(v), ms, gbps,
                    pct);
    }
    return 0;
}
