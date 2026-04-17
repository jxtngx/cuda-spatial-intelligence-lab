---
name: nsight-profiling
description: The lab's standard Nsight Systems + Nsight Compute workflow for any CUDA kernel claiming a performance number. Use after a kernel passes correctness tests and before any "this is fast" claim is made.
---

# Standard Nsight workflow

Every kernel that claims a performance number ships with **two** artifacts in
`report/`: a Nsight Systems `.qdrep` and a Nsight Compute `.ncu-rep`. No
artifact, no claim.

## Tool roles

- **Nsight Systems (`nsys`)** — *timeline*. Are kernels overlapping with
  copies? Are streams parallel? Are there idle gaps? CPU-side stalls?
- **Nsight Compute (`ncu`)** — *per-kernel SM-level metrics*. Where is
  time going inside the kernel? What's the bottleneck?
- **`compute-sanitizer`** — correctness gatekeeper. Run before either
  profiler.

## Step 0 — Sanitize

```bash
compute-sanitizer --tool memcheck   ./bench/bench_<name>
compute-sanitizer --tool racecheck  ./bench/bench_<name>
compute-sanitizer --tool synccheck  ./bench/bench_<name>
compute-sanitizer --tool initcheck  ./bench/bench_<name>
```

If any fire, fix first. Profile data from a buggy kernel is a lie.

## Step 1 — Coarse timeline (`nsys`)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --output=report/nsys_<name> \
  ./bench/bench_<name>
nsys stats report/nsys_<name>.qdrep > report/nsys_<name>_summary.txt
```

Things to look for:
1. Idle gaps between kernels — launch overhead or host stalls.
2. H↔D copies on a hot path (should be ~zero on Spark unified memory).
3. Streams serializing when they should be parallel.
4. `cudaMalloc`/`cudaFree` in a hot path (use a pool / `cudaMallocAsync`).

NVTX-instrument the C++ so the timeline reads like a story:

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePushA("tile_load");   /* ... */ nvtxRangePop();
nvtxRangePushA("compute");     /* ... */ nvtxRangePop();
```

## Step 2 — Per-kernel deep dive (`ncu`)

```bash
ncu --set full \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --section SourceCounters \
    --import-source on \
    -k <kernel_name> -c 5 \
    -o report/ncu_<name> \
    ./bench/bench_<name>
```

Open in `ncu-ui` or summarize via `ncu --import report/ncu_<name>.ncu-rep
--print-summary per-kernel`.

## Step 3 — Read the report in this order

1. **Speed of Light** — what's the bottleneck?
   - SM% high, mem% low → compute-bound.
   - mem% high, SM% low → memory-bound.
   - Both low → latency-bound (occupancy or sync stalls).
2. **Roofline** — where does this kernel sit vs the FP32/BF16/FP8
   ceilings for sm_121?
3. **Occupancy** — *theoretical* vs *achieved*. Big gap means you're
   limited by registers/shared-mem/block-size; pick the limiter and
   address it.
4. **Warp State Stats** — what are warps stalling on?
   - `Stall LG Throttle` → long-scoreboard / global memory latency.
   - `Stall MIO Throttle` → memory I/O pipe contention.
   - `Stall Wait` → barrier / `__syncthreads`.
   - `Stall Short Scoreboard` → shared-mem latency.
   - `Stall Tex Throttle` → texture / read-only cache pressure.
5. **Memory Workload Analysis** — L1/L2 hit rates, achieved DRAM
   bandwidth vs peak (~273 GB/s on Spark).
6. **Source Counters** — which lines burn cycles? Map to SASS via
   `--import-source on`.

## Step 4 — Hypothesis → one fix → re-profile

Discipline: **change one variable per iteration**. Tile size, vectorized
loads, async copies, register pressure — one per commit, with a fresh
report. Diff the reports.

## Roofline targets on sm_121

| Workload | Reasonable target |
|---|---|
| GEMM (FP32, large) | ≥ 70% of cuBLAS |
| GEMM (BF16/FP16 tensor core) | ≥ 60% of cuBLAS |
| GEMM (FP8 tensor core) | ≥ 50% of cuBLAS |
| Reduction | ≥ 80% of `cub::DeviceReduce` |
| Memcpy-shaped kernel | ≥ 85% of peak DRAM BW |
| Conv (im2col + GEMM) | ≥ 60% of cuDNN |
| Fused attention | ≥ 60% of FlashAttention-3 |

## Templates

`report/<kernel>_profile.md` template:

```
# <kernel> profile

## Setup
- Hardware: DGX Spark, sm_121
- Software: CUDA <X>, driver <Y>
- Workload: <shape, dtype, repetitions>

## Speed of Light
- SM%: __  | Mem%: __
- Verdict: <compute | memory | latency>-bound

## Roofline
- Achieved: __ TFLOP/s @ __ GB/s
- Ceiling for this dtype: __ TFLOP/s
- Position: __ % of ceiling

## Occupancy
- Theoretical: __ %  | Achieved: __ %
- Limiter: <registers | shared mem | block size>

## Top warp stalls
1. __ (__ %)
2. __ (__ %)

## Hypothesis
<one sentence>

## Next change
<one concrete refactor>

## Expected effect
<which metric, by how much>
```
