# Lab 03 — Python bindings

JIT-loaded PyTorch custom ops for the Lab-03 reduction, scan, and
histogram kernels. Pattern A from
[`.cursor/skills/python-bindings/SKILL.md`](../../../.cursor/skills/python-bindings/SKILL.md).

## Prerequisites

- Python env with CUDA-enabled `torch` and `pytest`.
- The C++/CUDA sources in `../src/` must compile (the JIT loader
  invokes nvcc behind the scenes — no need to run CMake first).

## Smoke test

```bash
cd labs/lab-03-reduce-scan/python
python reduce_scan_ext.py
```

This builds the extension on first run, then runs `reduce v4` on a
1M-element tensor and prints the absolute error vs `torch.sum`.

## Full test suite

```bash
cd labs/lab-03-reduce-scan/python
pytest -v
```

Three groups:

- `test_reduce_numerics` — all five Harris stages vs `torch.sum`.
- `test_scan_numerics` — both Hillis-Steele and `cooperative_groups::scan`
  vs `torch.cumsum`, single block (n ≤ 1024).
- `test_histogram_numerics` — both versions vs `torch.bincount`.
- `test_overhead_bound` — coarse wrapper-overhead sanity check
  (the strict 5% rule lands in Lab 05 via the C++ bench JSON
  hand-off).

## Notes

- `reduce(x, version="v0".."v4")`, `reduce_cub(x)`, `scan(x,
  version="hillis_steele"|"coop_groups")`, `scan_cub(x)`,
  `histogram(x, version="global"|"shared_warp")`.
- The wrapper threads `at::cuda::getCurrentCUDAStream()` into every
  launcher, so `with torch.cuda.stream(s): reduce(x)` runs on `s`.
- `scan` rejects `n > 1024` by design: this lab's scan is single-block.
  Multi-block decoupled look-back lands in Lab 04 / 05.
