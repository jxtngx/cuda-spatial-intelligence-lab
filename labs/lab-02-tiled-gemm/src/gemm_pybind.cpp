// PyTorch custom-op binding for gemm_tiled_async. This is the canonical
// shape from .cursor/skills/python-bindings/SKILL.md (Pattern A, JIT
// loader). The binding threads at::cuda::getCurrentCUDAStream() through
// the launcher so torch.cuda.Stream contexts work as expected. Do not
// modify — Weeks 3-16 copy this file as-is.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "gemm.hpp"

namespace {

torch::Tensor gemm_py(torch::Tensor A, torch::Tensor B,
                      double alpha, double beta,
                      torch::optional<torch::Tensor> C_opt,
                      int64_t version) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "A and B must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "A and B must be row-major contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    torch::Tensor C;
    if (C_opt.has_value()) {
        C = C_opt.value();
        TORCH_CHECK(C.is_cuda() && C.dtype() == torch::kFloat32 &&
                    C.is_contiguous(), "C must be float32 CUDA contiguous");
        TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
    } else {
        C = torch::zeros({M, N}, A.options());
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudalab::gemm::GemmShape shape{M, N, K};
    auto v = static_cast<cudalab::gemm::GemmVersion>(version);

    cudalab::gemm::gemm(shape,
                        static_cast<float>(alpha),
                        A.data_ptr<float>(),
                        B.data_ptr<float>(),
                        static_cast<float>(beta),
                        C.data_ptr<float>(),
                        stream.stream(),
                        v);
    return C;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_py,
          "Tiled GEMM (CUDA). version: 0=naive, 1=tiled32, 2=tiled64_padded, 3=tiled_async",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("alpha") = 1.0,
          pybind11::arg("beta") = 0.0,
          pybind11::arg("C") = pybind11::none(),
          pybind11::arg("version") = 3);
}
