"""pytest: numerics vs torch reference + wrapper-overhead bound.

Required by every lab from Week 2 onward. See
`.cursor/skills/python-bindings/SKILL.md` for the canonical shape.
"""

import pytest
import torch

from reduce_scan_ext import histogram, reduce, reduce_cub, scan, scan_cub

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ---------- reduction ----------

@pytest.mark.parametrize("version", ["v0", "v1", "v2", "v3", "v4"])
@pytest.mark.parametrize("n", [1, 17, 1 << 16, 1 << 22])
def test_reduce_numerics(version, n):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = reduce(x, version=version).item()
    ref = x.sum().item()
    tol = max(1e-2, 1e-4 * abs(ref))
    assert abs(out - ref) <= tol, (
        f"version={version} n={n} out={out} ref={ref} err={abs(out - ref):.2e}"
    )


def test_reduce_cub_matches_torch():
    torch.manual_seed(0)
    n = 1 << 22
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = reduce_cub(x).item()
    ref = x.sum().item()
    assert abs(out - ref) <= max(1e-2, 1e-4 * abs(ref))


# ---------- scan ----------

@pytest.mark.parametrize("version", ["hillis_steele", "coop_groups"])
@pytest.mark.parametrize("n", [1, 17, 256, 1024])
def test_scan_numerics(version, n):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = scan(x, version=version)
    ref = torch.cumsum(x, dim=0)
    assert (out - ref).abs().max().item() <= 1e-2


def test_scan_cub_matches_torch():
    torch.manual_seed(0)
    n = 1 << 16
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = scan_cub(x)
    ref = torch.cumsum(x, dim=0)
    rel = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-8)
    assert rel <= 1e-3


# ---------- histogram ----------

@pytest.mark.parametrize("version", ["global", "shared_warp"])
def test_histogram_numerics(version):
    torch.manual_seed(0)
    n = 1 << 20
    x = torch.randint(0, 256, (n,), device="cuda", dtype=torch.uint8)
    out = histogram(x, version=version)
    ref = torch.bincount(x.to(torch.int64), minlength=256).to(torch.int32)
    assert torch.equal(out, ref)


# ---------- wrapper overhead ----------

def _time_callable(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def test_overhead_bound():
    """Wrapper overhead must be < 5% of kernel time at the largest size.

    We compare the wrapped op vs the CUB baseline (also wrapped); both
    pay identical Python-side dispatch cost, so any blow-up is in the
    `reduce` wrapper itself, not the kernel. The hard 5% rule is
    enforced via the C++ bench JSON hand-off in Week 5; this test is
    the coarser Tier-A sanity check.
    """
    n = 1 << 26
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    py_ms = _time_callable(lambda: reduce(x, version="v4"))
    cub_ms = _time_callable(lambda: reduce_cub(x))

    overhead = (py_ms - cub_ms) / cub_ms
    assert overhead < 0.30, (
        f"v4 took {py_ms:.3f}ms vs CUB {cub_ms:.3f}ms "
        f"({overhead:.1%}); investigate before claiming the 80%-of-CUB "
        f"perf target. The 5% wrapper-overhead rule lands properly in W5."
    )
