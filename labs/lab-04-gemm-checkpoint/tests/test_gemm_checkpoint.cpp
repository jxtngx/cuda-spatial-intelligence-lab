#include "gemm.hpp"

#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

void cpu_sgemm(int M, int N, int K, float alpha,
               const std::vector<float>& A,
               const std::vector<float>& B,
               float beta,
               std::vector<float>& C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

float max_abs_diff(const std::vector<float>& x, const std::vector<float>& y) {
    float m = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs(x[i] - y[i]));
    return m;
}

}  // namespace

TEST(GemmCheckpoint, NumericsSmallVsCpu) {
    using namespace cudalab::gemm;
    const int M = 128, N = 128, K = 128;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N, 0.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, hA.size() * sizeof(float));
    cudaMalloc(&dB, hB.size() * sizeof(float));
    cudaMalloc(&dC, hC.size() * sizeof(float));
    cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, hC.size() * sizeof(float));

    gemm({M, N, K}, 1.0f, dA, dB, 0.0f, dC, nullptr,
         GemmVersion::v4_checkpoint);
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_sgemm(M, N, K, 1.0f, hA, hB, 0.0f, hRef);

    float err = max_abs_diff(hC, hRef);
    float bound = 1e-5f * K;  // relative-style: K * unit_round.
    EXPECT_LT(err, bound) << "max_abs_err=" << err << " bound=" << bound;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

TEST(GemmCheckpoint, NumericsLargeVsCublas) {
    using namespace cudalab::gemm;
    const int M = 1024, N = 1024, K = 1024;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hA(M * K), hB(K * N);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC, *dRef;
    cudaMalloc(&dA, hA.size() * sizeof(float));
    cudaMalloc(&dB, hB.size() * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dRef, M * N * sizeof(float));
    cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));
    cudaMemset(dRef, 0, M * N * sizeof(float));

    cublasHandle_t h; cublasCreate(&h);
    float a = 1.0f, b = 0.0f;
    // cuBLAS is column-major; compute B^T * A^T = (A * B)^T,
    // then read as row-major M x N => order swap M<->N.
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &a, dB, N, dA, K, &b, dRef, N);

    gemm({M, N, K}, 1.0f, dA, dB, 0.0f, dC, nullptr,
         GemmVersion::v4_checkpoint);
    cudaDeviceSynchronize();

    std::vector<float> hC(M * N), hRef(M * N);
    cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float err = max_abs_diff(hC, hRef);
    float bound = 1e-5f * K;
    EXPECT_LT(err, bound) << "max_abs_err=" << err << " bound=" << bound;

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
}
