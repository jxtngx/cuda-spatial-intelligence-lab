// Correctness tests for all four GEMM versions.
//   - Sizes 128, 512, 1024 are validated against a CPU reference triple
//     loop with max-abs-error tolerance.
//   - Size 4096 is validated against cublasSgemm with a relative
//     tolerance.

#include "gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using cudalab::gemm::gemm;
using cudalab::gemm::GemmShape;
using cudalab::gemm::GemmVersion;

namespace {

#define CUDA_OK(expr) ASSERT_EQ((expr), cudaSuccess)
#define CUBLAS_OK(expr) ASSERT_EQ((expr), CUBLAS_STATUS_SUCCESS)

void cpu_gemm(int M, int N, int K, float alpha,
              const std::vector<float>& A,
              const std::vector<float>& B,
              float beta,
              std::vector<float>& C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

class GemmCpuRef : public ::testing::TestWithParam<std::tuple<int, GemmVersion>> {};

TEST_P(GemmCpuRef, MaxAbsErrorWithinTol) {
    const auto [n, version] = GetParam();
    const int M = n, N = n, K = n;
    const float alpha = 1.5f, beta = 0.25f;

    std::vector<float> hA(M * K), hB(K * N), hC0(M * N), hC_ref(M * N);
    fill_random(hA, 1);
    fill_random(hB, 2);
    fill_random(hC0, 3);
    hC_ref = hC0;
    cpu_gemm(M, N, K, alpha, hA, hB, beta, hC_ref);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_OK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dC, hC0.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dC, hC0.data(), hC0.size() * sizeof(float), cudaMemcpyHostToDevice));

    gemm({M, N, K}, alpha, dA, dB, beta, dC, /*stream=*/nullptr, version);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> hC(M * N);
    CUDA_OK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i)
        max_abs = std::max(max_abs, std::abs(hC[i] - hC_ref[i]));
    EXPECT_LE(max_abs, 1e-2f) << "n=" << n << " version=" << static_cast<int>(version);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, GemmCpuRef,
    ::testing::Combine(
        ::testing::Values(128, 512, 1024),
        ::testing::Values(GemmVersion::v0_naive,
                          GemmVersion::v1_tiled32,
                          GemmVersion::v2_tiled64_padded,
                          GemmVersion::v3_tiled_async)));

class GemmCublasRef : public ::testing::TestWithParam<GemmVersion> {};

TEST_P(GemmCublasRef, MatchesCublasAt4096) {
    const GemmVersion version = GetParam();
    const int M = 4096, N = 4096, K = 4096;
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> hA(M * K), hB(K * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;
    CUDA_OK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_OK(cudaMalloc(&dC_ref, M * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dC, 0, M * N * sizeof(float)));
    CUDA_OK(cudaMemset(dC_ref, 0, M * N * sizeof(float)));

    // Reference: cuBLAS column-major. We pass our row-major matrices as
    // their transposes by swapping operands and dimensions.
    cublasHandle_t handle{};
    CUBLAS_OK(cublasCreate(&handle));
    CUBLAS_OK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K,
                          &alpha,
                          dB, N,
                          dA, K,
                          &beta,
                          dC_ref, N));

    gemm({M, N, K}, alpha, dA, dB, beta, dC, nullptr, version);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> hC(M * N), hC_ref(M * N);
    CUDA_OK(cudaMemcpy(hC.data(),     dC,     hC.size()     * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hC_ref.data(), dC_ref, hC_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, max_ref = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(hC[i] - hC_ref[i]));
        max_ref = std::max(max_ref, std::abs(hC_ref[i]));
    }
    const float rel = max_abs / std::max(max_ref, 1e-8f);
    EXPECT_LE(rel, 1e-3f) << "version=" << static_cast<int>(version);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
}

INSTANTIATE_TEST_SUITE_P(
    AllVersions, GemmCublasRef,
    ::testing::Values(GemmVersion::v0_naive,
                      GemmVersion::v1_tiled32,
                      GemmVersion::v2_tiled64_padded,
                      GemmVersion::v3_tiled_async));

}  // namespace
