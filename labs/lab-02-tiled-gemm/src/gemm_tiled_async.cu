// v3_tiled_async — same shape as v2 but with cuda::memcpy_async + a
// two-stage cuda::pipeline so the next K-strip is being copied from
// global -> shared while threads are multiplying the current strip.
//
// The pipeline + barrier scaffolding is provided. The two TODOs are:
//   (a) issue the next-stage memcpy_async BEFORE the multiply on the
//       current stage,
//   (b) wait/release the pipeline at the right places.

#include "gemm.hpp"

#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

namespace cudalab::gemm {

namespace {

constexpr int BM   = 64;
constexpr int BN   = 64;
constexpr int BK   = 16;
constexpr int TM   = 4;
constexpr int TN   = 4;
constexpr int NTY  = BM / TM;
constexpr int NTX  = BN / TN;
constexpr int NT   = NTY * NTX;
constexpr int PAD  = 1;  // we use the right padding here on purpose
constexpr int STAGES = 2;

__global__ void gemm_tiled_async_kernel(int M, int N, int K,
                                        float alpha,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float beta,
                                        float* __restrict__ C) {
    __shared__ float As[STAGES][BM][BK + PAD];
    __shared__ float Bs[STAGES][BK][BN + PAD];

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto pipe = cuda::make_pipeline(block, &pss);

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

    auto stage_copy = [&](int t, int stage) {
        pipe.producer_acquire();
        // A tile
        #pragma unroll
        for (int load = 0; load < (BM * BK) / NT; ++load) {
            int idx = load * NT + tid;
            int r = idx / BK;
            int c = idx % BK;
            int a_row = block_row + r;
            int a_col = t * BK + c;
            float v = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
            cuda::memcpy_async(&As[stage][r][c], &v, sizeof(float), pipe);
        }
        // B tile
        #pragma unroll
        for (int load = 0; load < (BK * BN) / NT; ++load) {
            int idx = load * NT + tid;
            int r = idx / BN;
            int c = idx % BN;
            int b_row = t * BK + r;
            int b_col = block_col + c;
            float v = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            cuda::memcpy_async(&Bs[stage][r][c], &v, sizeof(float), pipe);
        }
        pipe.producer_commit();
    };

    // Prime the pipeline with the first tile.
    if (num_tiles > 0) {
        stage_copy(0, 0);
    }

    for (int t = 0; t < num_tiles; ++t) {
        const int stage      = t % STAGES;
        const int next_t     = t + 1;
        const int next_stage = next_t % STAGES;

        // TODO(student) (a): if next_t < num_tiles, issue stage_copy for
        // the *next* tile here, BEFORE we wait on / consume the current
        // tile. This is what creates the copy/compute overlap.
        //
        //   if (next_t < num_tiles) {
        //       stage_copy(next_t, next_stage);
        //   }

        // TODO(student) (b): wait for the current stage's copies to land,
        // do the multiply, then release the stage so the producer can
        // refill it.
        //
        //   pipe.consumer_wait();
        //   __syncthreads();
        //
        //   #pragma unroll
        //   for (int k = 0; k < BK; ++k) {
        //       float a_reg[TM], b_reg[TN];
        //       #pragma unroll
        //       for (int i = 0; i < TM; ++i)
        //           a_reg[i] = As[stage][threadIdx.y * TM + i][k];
        //       #pragma unroll
        //       for (int j = 0; j < TN; ++j)
        //           b_reg[j] = Bs[stage][k][threadIdx.x * TN + j];
        //       #pragma unroll
        //       for (int i = 0; i < TM; ++i)
        //           #pragma unroll
        //           for (int j = 0; j < TN; ++j)
        //               acc[i][j] += a_reg[i] * b_reg[j];
        //   }
        //
        //   __syncthreads();
        //   pipe.consumer_release();

        (void)stage;
        (void)next_stage;
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

void launch_gemm_tiled_async(const GemmShape& s, float alpha,
                             const float* A, const float* B, float beta, float* C,
                             cudaStream_t stream) {
    dim3 block(NTX, NTY, 1);
    dim3 grid((s.N + BN - 1) / BN, (s.M + BM - 1) / BM, 1);
    gemm_tiled_async_kernel<<<grid, block, 0, stream>>>(s.M, s.N, s.K,
                                                        alpha, A, B, beta, C);
}

}  // namespace cudalab::gemm
