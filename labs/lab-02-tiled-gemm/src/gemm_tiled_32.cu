// v1_tiled32 — 32x32 shared-memory tile. Each block of 32x32 threads
// cooperatively loads a tile of A and B into shared memory and walks the
// K dimension one tile at a time. No padding yet — bank conflicts are
// expected and you should see them in the Nsight "Memory Workload
// Analysis" / "Shared Memory" sections.

#include "gemm.hpp"

#include <cuda_runtime.h>

namespace cudalab::gemm {

namespace {

constexpr int TILE = 32;

__global__ void gemm_tiled_32_kernel(int M, int N, int K,
                                     float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta,
                                     float* __restrict__ C) {
    // TODO(student) (a): declare the 32x32 shared tiles for A and B.
    //   __shared__ float As[TILE][TILE];
    //   __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        // TODO(student) (b): load one element of A and one element of B
        // into the shared tiles, guarded by bounds, then __syncthreads()
        // before the multiply, then __syncthreads() after the multiply
        // and before loading the next tile.
        //
        //   As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
        //       ? A[row * K + a_col] : 0.0f;
        //   Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
        //       ? B[b_row * N + col] : 0.0f;
        //   __syncthreads();
        //
        //   for (int k = 0; k < TILE; ++k)
        //       acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        //
        //   __syncthreads();
    }

    if (row < M && col < N) {
        // TODO(student) (c): write the epilogue
        //   C[row * N + col] = alpha * acc + beta * C[row * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

}  // namespace

void launch_gemm_tiled_32(const GemmShape& s, float alpha,
                          const float* A, const float* B, float beta, float* C,
                          cudaStream_t stream) {
    dim3 block(TILE, TILE, 1);
    dim3 grid((s.N + TILE - 1) / TILE, (s.M + TILE - 1) / TILE, 1);
    gemm_tiled_32_kernel<<<grid, block, 0, stream>>>(s.M, s.N, s.K,
                                                     alpha, A, B, beta, C);
}

}  // namespace cudalab::gemm
