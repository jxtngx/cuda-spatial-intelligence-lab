#pragma once

// Single-precision row-major SGEMM, Week-04 checkpoint extension of
// the Week-02 dispatcher.
//
//   C = alpha * A @ B + beta * C
//
// All matrices row-major float, device-resident. The Week-04
// addition is `v4_checkpoint`: register-tiled, float4-loaded,
// __ldg-hinted, double-buffered tiled GEMM, sweepable across
// (BM, BN, BK) ∈ {32, 64, 128}.
//
// Strategy-as-enum (Iglberger Ch 5) is reused unchanged so the
// Week-02 ABI survives.

#include <cstddef>
#include <cuda_runtime.h>

namespace cudalab::gemm {

enum class GemmVersion {
    v0_naive,
    v1_tiled32,
    v2_tiled64_padded,
    v3_tiled_async,
    v4_checkpoint,
};

struct GemmShape {
    int M;
    int N;
    int K;
};

// Compile-time tile shape. Passed as a non-type template parameter
// (C++20 NTTP-of-class-type, P0732R2) to the v4 kernel so each tile
// size compiles to its own register-allocated instantiation.
struct TileConfig {
    int BM;
    int BN;
    int BK;
    int RM;  // per-thread register sub-tile rows
    int RN;  // per-thread register sub-tile cols
};

// Host-side launcher. Pointers must be device-resident, row-major.
// `stream` may be nullptr (default stream).
void gemm(const GemmShape&  shape,
          float             alpha,
          const float*      A,   // M x K
          const float*      B,   // K x N
          float             beta,
          float*            C,   // M x N
          cudaStream_t      stream,
          GemmVersion       version);

// v4-only entry point that selects a tile config at runtime via a
// small switch over the supported configs. See gemm_launch.cu.
void gemm_v4_checkpoint(const GemmShape& shape,
                        float            alpha,
                        const float*     A,
                        const float*     B,
                        float            beta,
                        float*           C,
                        cudaStream_t     stream,
                        TileConfig       cfg);

}  // namespace cudalab::gemm
