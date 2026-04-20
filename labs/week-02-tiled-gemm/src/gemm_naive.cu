// v0_naive — one thread = one output element. Reads from global memory K
// times per thread; no shared memory; no register tiling. The strawman
// against which v1/v2/v3 are measured.

#include "gemm.hpp"

#include <cuda_runtime.h>

namespace cudalab::gemm {

namespace {

__global__ void gemm_naive_kernel(int M, int N, int K,
                                  float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float beta,
                                  float* __restrict__ C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // TODO(student): write the inner-product loop.
    //   acc += A[row, k] * B[k, col]   for k in [0, K).
    //   Row-major: A[row, k] = A[row * K + k]; B[k, col] = B[k * N + col].
    //
    // for (int k = 0; k < K; ++k) {
    //     acc += ... * ...;
    // }

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

}  // namespace

void launch_gemm_naive(const GemmShape& s, float alpha,
                       const float* A, const float* B, float beta, float* C,
                       cudaStream_t stream) {
    constexpr int BX = 16;
    constexpr int BY = 16;
    dim3 block(BX, BY, 1);
    dim3 grid((s.N + BX - 1) / BX, (s.M + BY - 1) / BY, 1);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(s.M, s.N, s.K,
                                                  alpha, A, B, beta, C);
}

}  // namespace cudalab::gemm
