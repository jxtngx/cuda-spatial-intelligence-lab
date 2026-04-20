// v2_tiled64_padded — 64x64 tile, padded shared memory, 4x4 register tile
// per thread. 16x16 = 256 threads per block; each thread owns a 4x4
// sub-tile of C in registers.

#include "gemm.hpp"

#include <cuda_runtime.h>

namespace cudalab::gemm {

namespace {

constexpr int BM   = 64;   // rows of C per block
constexpr int BN   = 64;   // cols of C per block
constexpr int BK   = 16;   // K-strip width per outer iteration
constexpr int TM   = 4;    // rows of C per thread (register tile)
constexpr int TN   = 4;    // cols of C per thread (register tile)
constexpr int NTY  = BM / TM;  // 16
constexpr int NTX  = BN / TN;  // 16
constexpr int NT   = NTY * NTX;  // 256 threads/block

// TODO(student) (a): choose PAD so that the row stride of `As`/`Bs` is
// not a multiple of 32, which is what eliminates 32-way bank conflicts on
// the 64-wide tile. PAD = 1 is the standard choice for 32-bit elements
// and 32 banks; verify with Nsight Compute "Shared Memory" before you
// commit. Replace the literal below.
constexpr int PAD  = 0;  // <-- TODO: replace 0 with the right padding

__global__ void gemm_tiled_64_kernel(int M, int N, int K,
                                     float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float beta,
                                     float* __restrict__ C) {
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const int tid = threadIdx.y * NTX + threadIdx.x;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Cooperative load: 256 threads load 64*16 = 1024 floats of A
        // and 16*64 = 1024 floats of B (4 elements per thread each).
        #pragma unroll
        for (int load = 0; load < (BM * BK) / NT; ++load) {
            int idx = load * NT + tid;
            int r = idx / BK;
            int c = idx % BK;
            int a_row = block_row + r;
            int a_col = t * BK + c;
            As[r][c] = (a_row < M && a_col < K)
                ? A[a_row * K + a_col] : 0.0f;
        }
        #pragma unroll
        for (int load = 0; load < (BK * BN) / NT; ++load) {
            int idx = load * NT + tid;
            int r = idx / BN;
            int c = idx % BN;
            int b_row = t * BK + r;
            int b_col = block_col + c;
            Bs[r][c] = (b_row < K && b_col < N)
                ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // TODO(student) (b): fill the register-tile inner loop.
        //
        // For each k in [0, BK):
        //   load TM elements of A column k into a small register array
        //   (one per row this thread owns), TN elements of B row k into
        //   another, then do the outer product into acc[TM][TN].
        //
        //   #pragma unroll
        //   for (int k = 0; k < BK; ++k) {
        //       float a_reg[TM];
        //       float b_reg[TN];
        //       #pragma unroll
        //       for (int i = 0; i < TM; ++i)
        //           a_reg[i] = As[threadIdx.y * TM + i][k];
        //       #pragma unroll
        //       for (int j = 0; j < TN; ++j)
        //           b_reg[j] = Bs[k][threadIdx.x * TN + j];
        //       #pragma unroll
        //       for (int i = 0; i < TM; ++i)
        //           #pragma unroll
        //           for (int j = 0; j < TN; ++j)
        //               acc[i][j] += a_reg[i] * b_reg[j];
        //   }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + threadIdx.y * TM + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = block_col + threadIdx.x * TN + j;
            if (col >= N) continue;
            float* c_ptr = &C[row * N + col];
            *c_ptr = alpha * acc[i][j] + beta * (*c_ptr);
        }
    }
}

}  // namespace

void launch_gemm_tiled_64(const GemmShape& s, float alpha,
                          const float* A, const float* B, float beta, float* C,
                          cudaStream_t stream) {
    dim3 block(NTX, NTY, 1);  // 16 x 16 = 256
    dim3 grid((s.N + BN - 1) / BN, (s.M + BM - 1) / BM, 1);
    gemm_tiled_64_kernel<<<grid, block, 0, stream>>>(s.M, s.N, s.K,
                                                     alpha, A, B, beta, C);
}

}  // namespace cudalab::gemm
