// CUTLASS SIMT SGEMM, used purely as a sanity baseline alongside
// cublasSgemm. Compiled only when CUDALAB_HAVE_CUTLASS is defined
// (CMake sets this when CUTLASS_DIR is provided).

#include "gemm.hpp"

#ifdef CUDALAB_HAVE_CUTLASS

#include <cutlass/gemm/device/gemm.h>
#include <stdexcept>

namespace cudalab::gemm {

void cutlass_sgemm_baseline(const GemmShape& s, float alpha,
                            const float* A, const float* B,
                            float beta, float* Cout,
                            cudaStream_t stream) {
    // -------------------------------------------------------------
    // TODO(student): pick the cutlass::gemm::device::Gemm template
    // parameters that compile against your installed CUTLASS.
    // Suggested defaults (SIMT, row-major, float):
    //   element types = float, layouts = RowMajor for A, B, C,
    //   op class = cutlass::arch::OpClassSimt,
    //   arch tag  = cutlass::arch::Sm80 (Spark sm_121 falls back).
    // Leave threadblock / warp / instruction shapes at their
    // CUTLASS-default values for this baseline.
    // -------------------------------------------------------------
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    Gemm op;
    typename Gemm::Arguments args(
        {s.M, s.N, s.K},
        {A, s.K},
        {B, s.N},
        {Cout, s.N},
        {Cout, s.N},
        {alpha, beta});
    auto status = op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS baseline GEMM failed");
    }
}

}  // namespace cudalab::gemm

#endif  // CUDALAB_HAVE_CUTLASS
