# Week 04 — Python wrapper

JIT loader for the `v4_checkpoint` SGEMM kernel.

## Usage

```bash
# Smoke test (compiles on first run, then cached).
python gemm_checkpoint_ext.py

# Full pytest (numerics + < 5% wrapper-overhead bound).
pytest -q test_gemm_checkpoint.py
```

## Pattern

Pattern A (JIT `cpp_extension.load()`) from
[`.cursor/skills/python-bindings/SKILL.md`](../../../.cursor/skills/python-bindings/SKILL.md).
The extension lives next to the kernel in `../src/`; this directory
holds only the Python entry points.
