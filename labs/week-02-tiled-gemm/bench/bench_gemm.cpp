// Microbenchmark: time all four GEMM versions and cublasSgemm at
// M=N=K=4096, report ms/call, GFLOP/s, and percent of cuBLAS. The §3
// performance target (gemm_tiled_async >= 50% of cublasSgemm) is read
// from this file's output.

#include "gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

using cudalab::gemm::gemm;
using cudalab::gemm::GemmShape;
using cudalab::gemm::GemmVersion;

namespace {

#define CUDA_CHECK(expr) do {                                \
    cudaError_t e = (expr);                                  \
    if (e != cudaSuccess) {                                  \
        std::fprintf(stderr, "CUDA error: %s at %s:%d\n",    \
                     cudaGetErrorString(e), __FILE__, __LINE__); \
        std::abort();                                        \
    }                                                        \
} while (0)

float bench_kernel(GemmShape s, float* dA, float* dB, float* dC,
                   GemmVersion v, int trials) {
    cudaEvent_t e0{}, e1{};
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int i = 0; i < 5; ++i)
        gemm(s, 1.0f, dA, dB, 0.0f, dC, nullptr, v);
    cudaDeviceSynchronize();

    cudaEventRecord(e0);
    for (int i = 0; i < trials; ++i)
        gemm(s, 1.0f, dA, dB, 0.0f, dC, nullptr, v);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / static_cast<float>(trials);
}

float bench_cublas(GemmShape s, float* dA, float* dB, float* dC, int trials) {
    cublasHandle_t h{};
    cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < 5; ++i)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    s.N, s.M, s.K, &alpha, dB, s.N, dA, s.K, &beta, dC, s.N);
    cudaDeviceSynchronize();

    cudaEvent_t e0{}, e1{};
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < trials; ++i)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    s.N, s.M, s.K, &alpha, dB, s.N, dA, s.K, &beta, dC, s.N);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cublasDestroy(h);
    return ms / static_cast<float>(trials);
}

const char* name(GemmVersion v) {
    switch (v) {
        case GemmVersion::v0_naive:          return "v0_naive";
        case GemmVersion::v1_tiled32:        return "v1_tiled32";
        case GemmVersion::v2_tiled64_padded: return "v2_tiled64_padded";
        case GemmVersion::v3_tiled_async:    return "v3_tiled_async";
    }
    return "?";
}

}  // namespace

int main() {
    constexpr int M = 4096, N = 4096, K = 4096;
    GemmShape s{M, N, K};
    const double flops_per_call = 2.0 * static_cast<double>(M) *
                                  static_cast<double>(N) *
                                  static_cast<double>(K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));

    std::vector<float> ones(M * K, 1.0f / 1024.0f);
    CUDA_CHECK(cudaMemcpy(dA, ones.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, ones.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * M * N));

    constexpr int TRIALS = 25;

    std::printf("GEMM  M=N=K=%d  (single precision)\n", M);
    std::printf("%-22s | %10s | %12s | %12s\n",
                "version", "ms/call", "GFLOP/s", "% of cuBLAS");

    const float ms_cublas = bench_cublas(s, dA, dB, dC, TRIALS);
    const double gflops_cublas = flops_per_call / (ms_cublas * 1e-3) / 1e9;

    for (auto v : {GemmVersion::v0_naive, GemmVersion::v1_tiled32,
                   GemmVersion::v2_tiled64_padded, GemmVersion::v3_tiled_async}) {
        const float ms = bench_kernel(s, dA, dB, dC, v, TRIALS);
        const double gflops = flops_per_call / (ms * 1e-3) / 1e9;
        const double pct = 100.0 * gflops / gflops_cublas;
        std::printf("%-22s | %10.3f | %12.2f | %11.1f%%\n",
                    name(v), ms, gflops, pct);
    }
    std::printf("%-22s | %10.3f | %12.2f | %11.1f%%\n",
                "cublasSgemm (ref)", ms_cublas, gflops_cublas, 100.0);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
