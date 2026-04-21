// PyTorch custom-op binding for the Week-04 v4_checkpoint kernel.
// JIT-loaded from python/gemm_checkpoint_ext.py (Pattern A from
// .cursor/skills/python-bindings/SKILL.md).

#include "gemm.hpp"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cudalab::gemm;

torch::Tensor gemm_checkpoint(torch::Tensor A, torch::Tensor B,
                              double alpha, double beta,
                              c10::optional<torch::Tensor> C_opt,
                              int64_t bm) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A,B must be contiguous");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2-D tensors only");
    TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch");

    const int M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = C_opt.has_value()
        ? C_opt.value()
        : torch::zeros({M, N}, A.options());
    TORCH_CHECK(C.is_cuda() && C.is_contiguous() && C.dtype() == torch::kFloat32);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");

    cg::TileConfig cfg = (bm == 32)
        ? cg::TileConfig{32, 32, 16, 4, 4}
        : (bm == 128)
            ? cg::TileConfig{128, 128, 16, 8, 8}
            : cg::TileConfig{64, 64, 16, 4, 4};

    auto stream = at::cuda::getCurrentCUDAStream();
    cg::gemm_v4_checkpoint({M, N, K},
                           static_cast<float>(alpha),
                           A.data_ptr<float>(),
                           B.data_ptr<float>(),
                           static_cast<float>(beta),
                           C.data_ptr<float>(),
                           stream.stream(),
                           cfg);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_checkpoint", &gemm_checkpoint,
          "Week-04 v4_checkpoint SGEMM (CUDA)",
          pybind11::arg("A"), pybind11::arg("B"),
          pybind11::arg("alpha") = 1.0,
          pybind11::arg("beta")  = 0.0,
          pybind11::arg("C")     = pybind11::none(),
          pybind11::arg("bm")    = 64);
}
