// Week-04 dispatcher. Routes GemmVersion::v4_checkpoint to the
// register-tiled kernel and selects launch geometry based on the
// requested TileConfig. Tier A: launch geometry for the 128-tile
// instantiation has a TODO for the student to fill in.

#include "gemm.hpp"

#include <stdexcept>

namespace cudalab::gemm {

template <TileConfig C>
extern __global__ void gemm_v4_kernel(int, int, int, float,
                                      const float*, const float*,
                                      float, float*);

namespace {

template <TileConfig C>
void launch_v4(const GemmShape& s, float alpha, const float* A,
               const float* B, float beta, float* Cout,
               cudaStream_t stream) {
    constexpr int TX = C.BN / C.RN;
    constexpr int TY = C.BM / C.RM;
    dim3 block(TX, TY, 1);
    dim3 grid((s.N + C.BN - 1) / C.BN,
              (s.M + C.BM - 1) / C.BM,
              1);
    gemm_v4_kernel<C><<<grid, block, 0, stream>>>(
        s.M, s.N, s.K, alpha, A, B, beta, Cout);
}

}  // namespace

void gemm_v4_checkpoint(const GemmShape& shape,
                        float alpha,
                        const float* A, const float* B,
                        float beta, float* Cout,
                        cudaStream_t stream,
                        TileConfig cfg) {
    if (cfg.BM == 32 && cfg.BN == 32 && cfg.BK == 16) {
        launch_v4<TileConfig{32, 32, 16, 4, 4}>(
            shape, alpha, A, B, beta, Cout, stream);
        return;
    }
    if (cfg.BM == 64 && cfg.BN == 64 && cfg.BK == 16) {
        launch_v4<TileConfig{64, 64, 16, 4, 4}>(
            shape, alpha, A, B, beta, Cout, stream);
        return;
    }
    if (cfg.BM == 128 && cfg.BN == 128 && cfg.BK == 16) {
        // -------------------------------------------------------------
        // TODO(student): pick the launch geometry for the 128-tile
        // instantiation. The default RM=RN=8 already picked above
        // gives a 16x16 thread block (256 threads). Justify this in
        // §6 Method — would 8x32 or 32x8 be better given Spark's
        // 4-way SM partition? Try one alternative and report.
        // -------------------------------------------------------------
        launch_v4<TileConfig{128, 128, 16, 8, 8}>(
            shape, alpha, A, B, beta, Cout, stream);
        return;
    }
    throw std::invalid_argument("gemm_v4_checkpoint: unsupported TileConfig");
}

void gemm(const GemmShape& shape, float alpha,
          const float* A, const float* B,
          float beta, float* Cout,
          cudaStream_t stream, GemmVersion version) {
    if (version == GemmVersion::v4_checkpoint) {
        gemm_v4_checkpoint(shape, alpha, A, B, beta, Cout, stream,
                           TileConfig{64, 64, 16, 4, 4});
        return;
    }
    throw std::invalid_argument(
        "Week-04 gemm() only implements v4_checkpoint; "
        "older versions live in labs/lab-02-tiled-gemm/");
}

}  // namespace cudalab::gemm
