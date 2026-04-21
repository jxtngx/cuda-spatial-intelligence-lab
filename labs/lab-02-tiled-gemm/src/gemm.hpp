#pragma once

// Single-precision row-major GEMM:  C = alpha * A @ B + beta * C
// A is M x K, B is K x N, C is M x N. All row-major, all float.
//
// Four implementations live in src/gemm_*.cu:
//   v0_naive          — one thread per output element, global memory only
//   v1_tiled32        — 32x32 shared-memory tile
//   v2_tiled64_padded — 64x64 tile, padded for bank conflicts, 4x4 register tile
//   v3_tiled_async    — v2 + cuda::memcpy_async + cuda::pipeline (double-buffered)
//
// The host-side dispatcher `gemm(...)` selects a version via the
// GemmVersion enum (Iglberger Ch 5: strategy-as-enum).

#include <cstddef>
#include <cuda_runtime.h>

namespace cudalab::gemm {

enum class GemmVersion {
    v0_naive,
    v1_tiled32,
    v2_tiled64_padded,
    v3_tiled_async,
};

struct GemmShape {
    int M;
    int N;
    int K;
};

// Host-side launcher. Pointers must be device-resident, row-major.
// `stream` may be nullptr (default stream).
void gemm(const GemmShape&  shape,
          float             alpha,
          const float*      A,   // M x K, row-major
          const float*      B,   // K x N, row-major
          float             beta,
          float*            C,   // M x N, row-major
          cudaStream_t      stream,
          GemmVersion       version);

}  // namespace cudalab::gemm
