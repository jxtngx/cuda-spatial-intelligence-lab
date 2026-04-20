// PyTorch JIT extension for Week-03 reduce / scan / histogram.
// Pattern A (PYBIND11_MODULE) per .cursor/skills/python-bindings/SKILL.md.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

#include "histogram.hpp"
#include "reduce.hpp"
#include "scan.hpp"

using week03::HistVersion;
using week03::ReduceVersion;
using week03::ScanVersion;

static torch::Tensor reduce_py(torch::Tensor x, int64_t version) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "float32 only");

    auto out = torch::empty({1}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    week03::launch_reduce(x.data_ptr<float>(), out.data_ptr<float>(),
                          static_cast<std::size_t>(x.numel()),
                          static_cast<ReduceVersion>(version), stream.stream());
    return out;
}

static torch::Tensor reduce_cub_py(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32);
    auto out = torch::empty({1}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    week03::launch_reduce_cub(x.data_ptr<float>(), out.data_ptr<float>(),
                              static_cast<std::size_t>(x.numel()), stream.stream());
    return out;
}

static torch::Tensor scan_py(torch::Tensor x, int64_t version) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32);
    TORCH_CHECK(x.numel() <= 1024, "Tier-A scan: n must be <= 1024");
    auto out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    week03::launch_scan(x.data_ptr<float>(), out.data_ptr<float>(),
                        static_cast<std::size_t>(x.numel()),
                        static_cast<ScanVersion>(version), stream.stream());
    return out;
}

static torch::Tensor scan_cub_py(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32);
    auto out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    week03::launch_scan_cub(x.data_ptr<float>(), out.data_ptr<float>(),
                            static_cast<std::size_t>(x.numel()), stream.stream());
    return out;
}

static torch::Tensor histogram_py(torch::Tensor x, int64_t version) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kUInt8);
    auto bins = torch::zeros({256}, x.options().dtype(torch::kInt32));
    auto stream = at::cuda::getCurrentCUDAStream();
    week03::launch_histogram(x.data_ptr<std::uint8_t>(),
                             reinterpret_cast<unsigned int*>(bins.data_ptr<int32_t>()),
                             static_cast<std::size_t>(x.numel()),
                             static_cast<HistVersion>(version), stream.stream());
    return bins;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce", &reduce_py, "sum-reduce float32 (version=0..4)");
    m.def("reduce_cub", &reduce_cub_py, "cub::DeviceReduce::Sum baseline");
    m.def("scan", &scan_py, "inclusive scan, single block, n<=1024 (version=0|1)");
    m.def("scan_cub", &scan_cub_py, "cub::DeviceScan::InclusiveSum baseline");
    m.def("histogram", &histogram_py, "256-bin uint8 histogram (version=0|1)");
}
