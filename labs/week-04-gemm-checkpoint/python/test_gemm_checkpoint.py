"""pytest for the Week-04 checkpoint GEMM Python wrapper.

Verifies:
  1. Numerics match torch.matmul to FP32 tolerance at M=N=K=1024.
  2. Wrapper overhead < 5% of kernel time at M=N=K=4096.

Pattern A (JIT) per `.cursor/skills/python-bindings/SKILL.md`.
"""

import time

import pytest
import torch

from gemm_checkpoint_ext import gemm


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@CUDA
def test_numerics_vs_torch_matmul():
    torch.manual_seed(0)
    M = N = K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = gemm(A, B, version="v4_checkpoint_bm64")
    ref = A @ B
    err = (C - ref).abs().max().item()
    bound = 1e-5 * K
    assert err < bound, f"max_abs_err={err:.3e} bound={bound:.3e}"


@CUDA
def test_overhead_bound():
    """Python-side wall-clock minus CUDA-event time must be < 5% of
    CUDA-event time at M=N=K=4096. This is the standing perf bar
    from `.cursor/skills/python-bindings/SKILL.md`."""
    M = N = K = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    for _ in range(5):
        gemm(A, B, version="v4_checkpoint_bm64")
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_ms = []
    py_ms = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.record()
        gemm(A, B, version="v4_checkpoint_bm64")
        e.record()
        e.synchronize()
        t1 = time.perf_counter()
        cuda_ms.append(s.elapsed_time(e))
        py_ms.append((t1 - t0) * 1e3)

    cuda_med = sorted(cuda_ms)[len(cuda_ms) // 2]
    py_med = sorted(py_ms)[len(py_ms) // 2]
    overhead = (py_med - cuda_med) / cuda_med
    assert overhead < 0.05, (
        f"wrapper overhead {overhead * 100:.2f}% exceeds 5% bound "
        f"(py={py_med:.3f}ms cuda={cuda_med:.3f}ms)"
    )
