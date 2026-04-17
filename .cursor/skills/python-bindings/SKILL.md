---
name: python-bindings
description: Canonical pattern for wrapping our CUDA kernels as PyTorch custom ops via torch.utils.cpp_extension. Use proactively when a lab from Week 2 onward needs a python/ wrapper, or when a production kernel needs to survive torch.compile / TensorRT export.
---

# Python bindings for CUDA kernels (`torch.utils.cpp_extension`)

Every weekly lab from Week 2 to Week 16 must expose its primary
kernel to Python as a PyTorch custom op. This is non-negotiable; the
goal is to make the kernel reachable from the Month 4 application
stack without rewriting anything. We use `torch.utils.cpp_extension`
(not raw pybind11, not nanobind) because it integrates with the
PyTorch tensor type, the CUDA stream, autograd, `torch.compile`, and
TensorRT export.

## Curriculum progression

| Phase     | Loader               | Op registration       | Why |
|-----------|----------------------|-----------------------|---|
| Month 1 (W2-4) | JIT `cpp_extension.load()` | `PYBIND11_MODULE`     | Fast iteration, no `setup.py` boilerplate. Numerics-first. |
| Month 2 (W5-8) | JIT `load()` still ok      | `PYBIND11_MODULE`     | Add stream-aware kwargs; thread `c10::cuda::getCurrentCUDAStream()` through. |
| Month 3 (W9-12)| AOT `setup.py` + `BuildExtension` | `TORCH_LIBRARY`       | The Cosmos fine-tune pipeline can't depend on JIT. Op must be visible to `torch.fx`. |
| Month 4 (W13-16)| AOT only            | `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` | Op must survive `torch.compile`, ONNX export, and TensorRT lowering in production. |

The mentor enforces this when scaffolding `python/` at each tier.

## File layout in every lab from Week 2 on

```
labs/week-NN-<slug>/
  python/
    setup.py             # AOT only (Month 3+); omit in Months 1-2
    <lab>_ext.py         # JIT loader OR AOT import + wrapper class
    test_<lab>.py        # pytest: numerics vs CPU reference + overhead bound
    README.md            # how to build and run
```

The C++/CUDA binding source lives next to the kernel, not in
`python/`:

```
labs/week-NN-<slug>/
  src/
    <kernel>.cu
    <kernel>.hpp
    <kernel>_pybind.cpp  # PYBIND11_MODULE or TORCH_LIBRARY here
```

This is so the kernel and its binding share a CMake target and
nothing about the binding is hidden from `/review-cuda`.

## Pattern A — JIT loader (Months 1-2)

`python/<lab>_ext.py`:

```python
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"

_ext = load(
    name="lab_ext",
    sources=[
        str(_SRC / "kernel.cu"),
        str(_SRC / "kernel_pybind.cpp"),
    ],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "-arch=sm_121", "-lineinfo"],
    verbose=False,
)

def kernel(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "tensors must be on CUDA"
    assert x.is_contiguous() and y.is_contiguous()
    assert x.dtype == y.dtype
    return _ext.kernel(x, y, alpha)
```

`src/kernel_pybind.cpp`:

```cpp
#include <torch/extension.h>
#include "kernel.hpp"

torch::Tensor kernel_py(torch::Tensor x, torch::Tensor y, double alpha) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(y.is_cuda(), "y must be on CUDA");
    TORCH_CHECK(x.dtype() == y.dtype(), "dtype mismatch");
    auto out = torch::empty_like(y);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_kernel(x.data_ptr(), y.data_ptr(), out.data_ptr(),
                  x.numel(), static_cast<float>(alpha), stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel_py, "lab kernel (CUDA)");
}
```

## Pattern B — AOT + `TORCH_LIBRARY` (Months 3-4)

`python/setup.py`:

```python
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SRC = Path(__file__).resolve().parent.parent / "src"

setup(
    name="lab_ext",
    ext_modules=[
        CUDAExtension(
            name="lab_ext",
            sources=[str(SRC / "kernel.cu"), str(SRC / "kernel_pybind.cpp")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", "-arch=sm_121", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

`src/kernel_pybind.cpp` (custom-op flavor):

```cpp
#include <torch/library.h>
#include "kernel.hpp"

torch::Tensor kernel_cuda(const torch::Tensor& x, const torch::Tensor& y, double alpha) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda());
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous());
    auto out = torch::empty_like(y);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_kernel(x.data_ptr(), y.data_ptr(), out.data_ptr(),
                  x.numel(), static_cast<float>(alpha), stream);
    return out;
}

TORCH_LIBRARY(lab, m) {
    m.def("kernel(Tensor x, Tensor y, float alpha) -> Tensor");
}
TORCH_LIBRARY_IMPL(lab, CUDA, m) {
    m.impl("kernel", &kernel_cuda);
}
```

Used from Python as `torch.ops.lab.kernel(x, y, alpha)` — visible to
`torch.compile`, exportable to ONNX, lowerable by TensorRT.

## Required `pytest` shape (every lab, every week from W2)

`python/test_<lab>.py`:

```python
import time
import pytest
import torch
import lab_ext  # JIT-loaded or installed

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 17, 1 << 16, 1 << 28])
def test_numerics(dtype, n):
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype, device="cuda")
    y = torch.randn(n, dtype=dtype, device="cuda")
    alpha = 1.5

    out = lab_ext.kernel(x, y, alpha)

    ref = (alpha * x.float() + y.float()).to(dtype)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}[dtype]
    assert (out.float() - ref.float()).abs().max().item() <= tol

def test_overhead_bound():
    """Wrapper overhead must be < 5% of kernel time at the largest size."""
    n = 1 << 28
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    iters = 50
    start.record()
    for _ in range(iters):
        lab_ext.kernel(x, y, 1.5)
    stop.record()
    torch.cuda.synchronize()
    py_ms = start.elapsed_time(stop) / iters

    kernel_ms = float(__import__("os").environ.get("KERNEL_MS", "0"))
    if kernel_ms > 0:
        overhead = (py_ms - kernel_ms) / kernel_ms
        assert overhead < 0.05, f"wrapper overhead {overhead:.1%} > 5%"
```

(Drop the `KERNEL_MS` env coupling once the C++ bench writes a JSON
file the test can read; that lands in Week 5.)

## Common pitfalls

- **Wrong stream.** Always thread
  `at::cuda::getCurrentCUDAStream()` into `launch_kernel`. Otherwise
  your op runs on the default stream and the user's `torch.cuda.Stream`
  context is silently ignored.
- **Implicit copies.** A non-contiguous input will be silently copied
  by some PyTorch ops. Assert `is_contiguous()` and let the caller
  fix it; never copy in the wrapper.
- **`double` arguments.** PyTorch passes Python `float` as C++
  `double`. Cast to `float` explicitly inside the wrapper if the
  kernel takes `float`.
- **JIT cache.** `cpp_extension.load()` caches by source hash. If you
  edit a header that's `#include`d from `.cu`, the cache may miss.
  Set `verbose=True` once to confirm a rebuild happened.
- **`__half` / `__nv_bfloat16` from torch.** Bridge via
  `torch::Half` / `torch::BFloat16` (or just dispatch on
  `x.scalar_type()` with `AT_DISPATCH_FLOATING_TYPES_AND2`).
- **`torch.compile` invisibility.** A `PYBIND11_MODULE` op is opaque
  to `torch.compile` and will cause graph breaks. Once you're in
  Month 3, switch to `TORCH_LIBRARY` so the op participates in the
  graph.
- **TensorRT export.** Only `TORCH_LIBRARY` ops with a registered
  meta kernel (shape-only `CompositeExplicitAutograd` impl) export
  cleanly. Add a meta impl in Month 4.

## Hand-offs

- **Performance regressions** in the wrapper → `cuda-perf-profiler`.
  Wrapper overhead is usually one of: an extra allocation, a stream
  mismatch, a non-contiguous copy.
- **Production export** (TRT, ONNX, Triton) → `model-deployer`.
- **Fine-tune integration** (custom op consumed by NeMo AutoModel) →
  `nemo-engineer`.
- **Agent-side use** (the NextJS DeepAgent calling a kernel via the
  served model) → `langchain-deepagents-architect`.

## Anti-patterns

- Writing raw `pybind11` modules that don't link `<torch/extension.h>`
  — you lose tensor interop and stream awareness.
- Using `ctypes` or `cffi` to call the kernel — you lose autograd
  and `torch.compile` integration entirely.
- Allocating output via `cudaMalloc` inside the wrapper — always
  use `torch::empty_like(...)` so PyTorch's allocator manages it.
- Skipping the `pytest` because "the C++ test already covers it" —
  the Python test is what catches stream and dtype-bridge bugs the
  C++ test cannot see.
