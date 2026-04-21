"""pytest: numerics vs torch.matmul + wrapper-overhead bound.

Required by every lab from Lab 02 onward. See
`.cursor/skills/python-bindings/SKILL.md` for the canonical shape.
"""

import pytest
import torch

from gemm_ext import gemm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.mark.parametrize("n", [128, 512, 1024])
def test_numerics_small(n):
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, n, device="cuda", dtype=torch.float32)

    out = gemm(A, B, version="tiled_async")
    ref = A @ B
    assert (out - ref).abs().max().item() <= 1e-2, "numerics failed at small N"


def test_numerics_4096():
    torch.manual_seed(0)
    n = 4096
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, n, device="cuda", dtype=torch.float32)

    out = gemm(A, B, version="tiled_async")
    ref = A @ B
    rel = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-8)
    assert rel <= 1e-3, f"relative error {rel:.2e} exceeds 1e-3 at 4096^3"


def _time_callable(fn, iters=20):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def test_overhead_bound():
    """Wrapper overhead must be < 5% of kernel time at M=N=K=4096.

    We compare the wrapped op to a "raw" timing where we still go through
    the same wrapper but call into the kernel; the difference between
    that and a same-stream torch.matmul on the same tensors gives us a
    bound on Python-side dispatch overhead. The actual kernel-only time
    will be measured via Nsight Compute and reported in LAB.md §7.
    """
    n = 4096
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, n, device="cuda", dtype=torch.float32)

    py_ms = _time_callable(lambda: gemm(A, B, version="tiled_async"))
    ref_ms = _time_callable(lambda: torch.matmul(A, B))

    overhead = (py_ms - ref_ms) / ref_ms
    assert overhead < 0.05 or py_ms < ref_ms, (
        f"wrapper overhead {overhead:.1%} > 5% (py={py_ms:.3f}ms, "
        f"ref={ref_ms:.3f}ms). Note: this comparison conflates kernel "
        f"speed with wrapper overhead; tighten via the C++ bench JSON "
        f"hand-off in Lab 05."
    )
