"""JIT loader + thin Python wrappers for Week-03 reduce / scan / histogram.

Pattern A (JIT) from `.cursor/skills/python-bindings/SKILL.md`. The
extension is compiled on first import and cached by source hash.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"

_ext = load(
    name="week03_reduce_scan_ext",
    sources=[
        str(_SRC / "reduce.cu"),
        str(_SRC / "reduce_cub.cu"),
        str(_SRC / "scan.cu"),
        str(_SRC / "scan_cub.cu"),
        str(_SRC / "histogram.cu"),
        str(_SRC / "reduce_scan_pybind.cpp"),
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


_REDUCE_VERSIONS = {"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4}
_SCAN_VERSIONS = {"hillis_steele": 0, "coop_groups": 1}
_HIST_VERSIONS = {"global": 0, "shared_warp": 1}


def reduce(x: torch.Tensor, version: str | int = "v4") -> torch.Tensor:
    """Sum-reduce a 1-D float32 CUDA tensor; returns a 1-element tensor."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    v = _REDUCE_VERSIONS[version] if isinstance(version, str) else int(version)
    return _ext.reduce(x, v)


def reduce_cub(x: torch.Tensor) -> torch.Tensor:
    """cub::DeviceReduce::Sum baseline."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    return _ext.reduce_cub(x)


def scan(x: torch.Tensor, version: str | int = "coop_groups") -> torch.Tensor:
    """Inclusive scan, single block (n <= 1024). Returns same shape/dtype."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    assert x.numel() <= 1024, "Tier-A scan: n must be <= 1024"
    v = _SCAN_VERSIONS[version] if isinstance(version, str) else int(version)
    return _ext.scan(x, v)


def scan_cub(x: torch.Tensor) -> torch.Tensor:
    """cub::DeviceScan::InclusiveSum baseline."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    return _ext.scan_cub(x)


def histogram(x: torch.Tensor, version: str | int = "shared_warp") -> torch.Tensor:
    """256-bin histogram of a 1-D uint8 CUDA tensor; returns int32[256]."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.uint8
    v = _HIST_VERSIONS[version] if isinstance(version, str) else int(version)
    return _ext.histogram(x, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    s = reduce(x, "v4").item()
    ref = x.sum().item()
    print(f"smoke ok, reduce v4 sum={s:.4f} ref={ref:.4f} err={abs(s - ref):.2e}")
