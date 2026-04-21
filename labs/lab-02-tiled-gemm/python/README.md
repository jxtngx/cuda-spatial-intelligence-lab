# Lab 02 — Python bindings

JIT-loaded PyTorch custom op for `gemm_tiled_async`. Pattern A from
[`.cursor/skills/python-bindings/SKILL.md`](../../../.cursor/skills/python-bindings/SKILL.md).

## Prerequisites

- Python env with CUDA-enabled `torch` and `pytest`.
- The C++/CUDA sources in `../src/` must compile (the JIT loader
  invokes nvcc behind the scenes — no need to run CMake first).

## Smoke test

```bash
cd labs/lab-02-tiled-gemm/python
python gemm_ext.py
```

This builds the extension on first run, then runs a 1024^3 GEMM and
prints the max-abs-error vs `torch.matmul`.

## Full test suite

```bash
cd labs/lab-02-tiled-gemm/python
pytest -v
```

Two test groups:

- `test_numerics_*` — numerics vs `torch.matmul` at sizes 128, 512,
  1024, 4096.
- `test_overhead_bound` — wrapper overhead < 5% of kernel time at
  M=N=K=4096.

## Notes

- The first import compiles the extension; subsequent runs hit the
  `cpp_extension` cache. If you edit a header included from a `.cu`
  file and the cache misses something, set `verbose=True` in
  `gemm_ext.py` to confirm a rebuild.
- The wrapper threads `c10::cuda::getCurrentCUDAStream()` into the
  launcher, so `with torch.cuda.stream(s): gemm(A, B)` runs on `s`.
- `version=` accepts `"naive" | "tiled32" | "tiled64" | "tiled_async"`
  (default `"tiled_async"`).
