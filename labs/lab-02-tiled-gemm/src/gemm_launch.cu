// Strategy-as-enum dispatcher (Iglberger Ch 5). The host launcher selects
// one of the four kernels by GemmVersion and forwards the launch.

#include "gemm.hpp"

#include <stdexcept>

namespace cudalab::gemm {

void launch_gemm_naive(const GemmShape&, float, const float*, const float*,
                       float, float*, cudaStream_t);
void launch_gemm_tiled_32(const GemmShape&, float, const float*, const float*,
                          float, float*, cudaStream_t);
void launch_gemm_tiled_64(const GemmShape&, float, const float*, const float*,
                          float, float*, cudaStream_t);
void launch_gemm_tiled_async(const GemmShape&, float, const float*, const float*,
                             float, float*, cudaStream_t);

void gemm(const GemmShape& shape, float alpha,
          const float* A, const float* B, float beta, float* C,
          cudaStream_t stream, GemmVersion version) {
    switch (version) {
        case GemmVersion::v0_naive:
            return launch_gemm_naive(shape, alpha, A, B, beta, C, stream);
        case GemmVersion::v1_tiled32:
            return launch_gemm_tiled_32(shape, alpha, A, B, beta, C, stream);
        case GemmVersion::v2_tiled64_padded:
            return launch_gemm_tiled_64(shape, alpha, A, B, beta, C, stream);
        case GemmVersion::v3_tiled_async:
            return launch_gemm_tiled_async(shape, alpha, A, B, beta, C, stream);
    }
    throw std::logic_error("unknown GemmVersion");
}

}  // namespace cudalab::gemm
