---
name: weekly-checkpoint
description: The 5-axis rubric and grading workflow used at the end of every week (and stricter in checkpoint weeks 4/8/12/16). Use when running /checkpoint or evaluating whether the user is ready to advance.
---

# Weekly checkpoint rubric

End of every week. The `curriculum-mentor` subagent uses this as its
grading sheet.

## The 5 axes

| Axis | What is graded |
|---|---|
| **Correctness** | Tests pass at all sizes incl. edge cases; max-error within stated tolerance |
| **Performance** | Achieves the lab's stated perf target on Spark sm_121 |
| **Idiom** | C++20 modern (concepts/ranges/RAII); CUDA modern (cooperative groups, async copies, tensor cores where relevant) |
| **Profile evidence** | Nsight Systems trace + Nsight Compute report committed to `report/`; bottleneck named in writeup |
| **Writeup** | `report/LAB.md` has hypothesis → method → results → next steps; cites primary sources |

## Scoring scale (per axis, 0-4)

- **4 — Excellent**: exceeds the bar in a way a reviewer would compliment.
- **3 — Solid**: hits the bar cleanly.
- **2 — Pass**: meets the bar with a few rough edges.
- **1 — Below**: fails one part of the bar.
- **0 — Fail**: missing or fundamentally wrong.

## Pass thresholds

| Week type | Total /20 needed |
|---|---|
| Regular weeks (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15) | **14** |
| Checkpoint weeks (4, 8, 12) | **17** |
| Capstone (16) | **18** |

If a week scores below threshold, the verdict is `REWORK`. The mentor
returns the 1-3 highest-leverage fixes; the user iterates and re-runs
`/checkpoint`.

## Per-axis detail

### Correctness

- Tests exist and pass: 2 minimum.
- Includes at least one non-power-of-two size: +1.
- Includes N=0 / N=1 edge cases: +1.
- `compute-sanitizer` clean (memcheck + racecheck + synccheck): +1
  (capped at 4).

### Performance

- Below 50% of target: 0.
- 50-79% of target: 1.
- 80-99% of target: 2.
- Hits target exactly (within 5%): 3.
- Exceeds target by ≥ 10% (or hits a stretch target named in `LAB.md`): 4.

### Idiom

C++:
- No raw `new`/`delete`: required.
- RAII for all CUDA handles: +1.
- Concepts on templated host helpers: +1.
- `std::span` / `std::mdspan` for views: +1.

CUDA:
- Coalesced global access: required.
- Cooperative groups or warp shuffles where reductions exist: +1.
- Async copies / `cuda::pipeline` where global→shared dominates: +1.
- Tensor cores for any matmul-shaped workload: +1 (only if relevant).

### Profile evidence

- Nsight Systems `.qdrep` committed: +1.
- Nsight Compute `.ncu-rep` committed for the hot kernel: +1.
- Bottleneck named in writeup with one of {compute, memory, latency,
  occupancy}: +1.
- Roofline position quoted (X% of FP32/BF16/FP8 ceiling): +1.

### Writeup

- `report/LAB.md` exists with sections: Spec, Method, Results, Next: +1.
- Hypothesis stated *before* the experiment: +1.
- Cites at least one chapter from PMPP/Iglberger/Stroustrup/Hartley/
  Szeliski/Torralba **with section number**: +1.
- "What I would do next" is concrete and actionable, not "make it faster": +1.

## Output template (mentor uses this)

```
# Week N checkpoint

## Scores
- Correctness:      X/4   — <one-line evidence>
- Performance:      X/4   — <achieved vs target>
- Idiom:            X/4   — <one specific call-out>
- Profile evidence: X/4   — <which artifacts present>
- Writeup:          X/4   — <quality observation>

Total: NN/20  (threshold: NN)

## Verdict
READY_TO_ADVANCE | REWORK | BLOCKED

## Highest-leverage fixes (if REWORK)
1. ...
2. ...
3. ...
```
