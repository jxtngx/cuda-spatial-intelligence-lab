"""JIT loader + thin Python wrapper for the Week-02 tiled GEMM.

Pattern A (JIT) from `.cursor/skills/python-bindings/SKILL.md`. The
extension is compiled on first import and cached by source hash.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"

_ext = load(
    name="week02_gemm_ext",
    sources=[
        str(_SRC / "gemm_naive.cu"),
        str(_SRC / "gemm_tiled_32.cu"),
        str(_SRC / "gemm_tiled_64.cu"),
        str(_SRC / "gemm_tiled_async.cu"),
        str(_SRC / "gemm_launch.cu"),
        str(_SRC / "gemm_pybind.cpp"),
    ],
    extra_include_paths=[str(_SRC)],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-arch=sm_121",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ],
    verbose=False,
)


_VERSION_MAP = {
    "naive": 0,
    "tiled32": 1,
    "tiled64": 2,
    "tiled_async": 3,
}


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: torch.Tensor | None = None,
    version: str | int = "tiled_async",
) -> torch.Tensor:
    """C = alpha * A @ B + beta * C, row-major float32 on CUDA."""
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.is_contiguous() and B.is_contiguous(), "A and B must be contiguous"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "float32 only"
    v = _VERSION_MAP[version] if isinstance(version, str) else int(version)
    return _ext.gemm(A, B, float(alpha), float(beta), C, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    M = N = K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = gemm(A, B, version="tiled_async")
    ref = A @ B
    err = (C - ref).abs().max().item()
    print(f"smoke ok, M=N=K={M}, max_abs_err={err:.3e}")
