// Week-04 checkpoint kernel: register-tiled, float4-vectorized,
// __ldg-hinted, double-buffered SGEMM.
//
// This file ships as a CORRECT-but-SLOW baseline so the rest of the
// lab (tests, bench, Python wrapper) is wired end-to-end on day one.
// Your job (Tier A) is to replace the four marked TODO blocks with
// the real optimizations and watch the bench number climb toward the
// 70%-of-cuBLAS pass line.

#include "gemm.hpp"

#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cudalab::gemm {

// Default tile shape used by tests; the launcher instantiates other
// shapes via the TileConfig NTTP below.
inline constexpr TileConfig kDefaultTile = {
    .BM = 64, .BN = 64, .BK = 16, .RM = 4, .RN = 4,
};

template <TileConfig C>
__global__ void gemm_v4_kernel(int M, int N, int K,
                               float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float beta,
                               float* __restrict__ Cout) {
    // Thread-block tile of C: BM x BN. Each thread owns an RM x RN
    // register sub-tile, so the block has (BM/RM) x (BN/RN) threads.
    static_assert(C.BM % C.RM == 0 && C.BN % C.RN == 0);
    constexpr int TX = C.BN / C.RN;
    constexpr int TY = C.BM / C.RM;

    const int block_row = blockIdx.y * C.BM;
    const int block_col = blockIdx.x * C.BN;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float As[C.BM][C.BK];
    __shared__ float Bs[C.BK][C.BN];

    float c_reg[C.RM][C.RN] = {};

    for (int k0 = 0; k0 < K; k0 += C.BK) {
        // -------------------------------------------------------------
        // TODO(student) #1: vectorized + read-only load of A tile
        //
        // Replace this scalar load with a float4 load (one float4 per
        // thread covers four columns of one row), and ensure the load
        // address is 16-byte aligned. Expect DRAM transactions to
        // roughly halve and `smsp__inst_executed_pipe_lsu` to drop.
        // -------------------------------------------------------------
        for (int i = ty; i < C.BM; i += TY) {
            for (int j = tx; j < C.BK; j += TX) {
                int r = block_row + i;
                int c = k0 + j;
                As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
        }

        // -------------------------------------------------------------
        // TODO(student) #2: vectorized + __ldg load of B tile
        //
        // Same idea as #1 but for B, and route the global load
        // through __ldg(...) (or rely on `const __restrict__` +
        // `--use_fast_math`-style heuristics). Watch the
        // *Memory Workload Analysis → L1/TEX Hit Rate* counter.
        // -------------------------------------------------------------
        for (int i = ty; i < C.BK; i += TY) {
            for (int j = tx; j < C.BN; j += TX) {
                int r = k0 + i;
                int c = block_col + j;
                Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
            }
        }
        __syncthreads();

        // -------------------------------------------------------------
        // TODO(student) #3: 4x4 register tile inner loop
        //
        // Replace this scalar accumulate with the canonical
        // register-tile pattern:
        //   float a_reg[RM]; float b_reg[RN];
        //   for kk in 0..BK:
        //     load a_reg[*] = As[ty*RM + r][kk]
        //     load b_reg[*] = Bs[kk][tx*RN + c]
        //     for r in 0..RM: for c in 0..RN: c_reg[r][c] += a_reg[r] * b_reg[c]
        // Expect: shared-memory traffic drops ~RM*RN x; register
        // pressure rises (watch `nvcc -Xptxas=-v`).
        // -------------------------------------------------------------
        for (int r = 0; r < C.RM; ++r) {
            for (int c = 0; c < C.RN; ++c) {
                int row = ty * C.RM + r;
                int col = tx * C.RN + c;
                float acc = 0.0f;
                for (int kk = 0; kk < C.BK; ++kk) {
                    acc += As[row][kk] * Bs[kk][col];
                }
                c_reg[r][c] += acc;
            }
        }
        __syncthreads();

        // -------------------------------------------------------------
        // TODO(student) #4: double-buffer the As/Bs loads
        //
        // Add a second shared-memory buffer (As[2][BM][BK],
        // Bs[2][BK][BN]) and a `cuda::pipeline<thread_scope_block, 2>`
        // so tile k+1 is being copied while tile k is multiplied.
        // Expect *Warp State → Stall LongScoreboard* to drop.
        // Reference: libcu++ <cuda/pipeline>; CUTLASS GEMM tutorial.
        // -------------------------------------------------------------
    }

    // Epilogue: scale + accumulate into C.
    for (int r = 0; r < C.RM; ++r) {
        for (int c = 0; c < C.RN; ++c) {
            int row = block_row + ty * C.RM + r;
            int col = block_col + tx * C.RN + c;
            if (row < M && col < N) {
                float old = (beta != 0.0f) ? Cout[row * N + col] : 0.0f;
                Cout[row * N + col] = alpha * c_reg[r][c] + beta * old;
            }
        }
    }
}

// Explicit instantiations for the three tile shapes the launcher
// supports. Add more here if you sweep further.
template __global__ void gemm_v4_kernel<TileConfig{32, 32, 16, 4, 4}>(
    int, int, int, float, const float*, const float*, float, float*);
template __global__ void gemm_v4_kernel<TileConfig{64, 64, 16, 4, 4}>(
    int, int, int, float, const float*, const float*, float, float*);
template __global__ void gemm_v4_kernel<TileConfig{128, 128, 16, 8, 8}>(
    int, int, int, float, const float*, const float*, float, float*);

}  // namespace cudalab::gemm
