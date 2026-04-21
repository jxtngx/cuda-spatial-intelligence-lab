---
name: lab-notebook
description: How to author a rigorous report/LAB.md for a curriculum lab — the writeup that is graded on the Writeup axis of the rubric and shipped publicly with the repo. Use when running /lab-report or finishing a lab.
---

# Lab notebook (`report/LAB.md`) authoring

Every `labs/lab-NN-*/` ships a `report/LAB.md`. It is the write-up
recruiters will read. Treat it as a mini-paper.

## Required sections

```markdown
# Lab N — <Title>

## 1. Spec
- Inputs / outputs (shapes, dtypes, layouts)
- Numerical tolerance
- Performance target (cite the curriculum file)
- Constraints

## 2. Hypothesis
*Written before any optimization.* What do you expect the bottleneck to
be? Why? Which Nsight metric will move first?

## 3. Method
For each version (v0 naive → vN optimized), one paragraph:
- What changed.
- Why (cite PMPP §, Iglberger guideline, paper).
- One sentence on what you expected.

Include any non-obvious build flags.

## 4. Results
A table. Always. With units.

| Version | Time (ms) | GFLOP/s | % of <baseline> | Achieved BW (GB/s) |
|---|---|---|---|---|
| v0 naive | ... | ... | ... | ... |
| v1 tiled | ... | ... | ... | ... |
| ...     | ... | ... | ... | ... |

Reference Nsight reports inline:
> Roofline analysis: see `report/ncu_<name>_v3.ncu-rep` — kernel sits at
> 73% of FP32 ceiling, memory-bound on L2.

## 5. Discussion
- Where did the predicted bottleneck NOT show up? Why?
- What surprised you?
- What did you learn that you'll carry forward?

## 6. What I would do next
At least one concrete next experiment (with the metric you'd watch).
"Try TMA + thread-block clusters and see if SOL Memory drops below 60%."

## 7. References
- PMPP 4e Ch X §Y.Z
- Paper title (Author et al, year, arXiv:XXXX.XXXXX)
- Curriculum file: `.cursor/skills/curriculum-plan/month-N-*.md`
```

## Tone rules

- **Hypothesis before experiment.** A report that opens with results
  and back-fills the hypothesis loses a point on Writeup.
- **Cite specifically.** "PMPP" is not a citation. "PMPP 4e §5.4" is.
- **Show the table, then explain.** Numbers first, prose second.
- **One concrete next experiment.** "Make it faster" is not a next step.

## Anti-patterns

- "I did the lab; it works." — that's not a report.
- Screenshots of Nsight without quoting the metric. Quote the metric;
  link the report file.
- Conclusions that are also the hypothesis.
- Dates and machine names in the body. Put those in `## Setup` only.

## Ready-to-paste skeleton

```markdown
# Lab 02 — Tiled SGEMM

## Setup
- Hardware: DGX Spark, sm_121, driver <X>, CUDA <Y>
- Toolchain: GCC 13, CMake 3.28, Ninja
- Reference: cuBLAS sgemm
- Workload: M=N=K=4096 single precision, 100 trials

## 1. Spec
...

## 2. Hypothesis
...

## 3. Method
### v0 — naive
### v1 — 32×32 tiled
### v2 — 64×64 tiled, padded
### v3 — async-copy double-buffered

## 4. Results
| Version | Time (ms) | GFLOP/s | % cuBLAS |
|---|---|---|---|
| v0 | ... | ... | ... |
| v1 | ... | ... | ... |
| v2 | ... | ... | ... |
| v3 | ... | ... | ... |

## 5. Discussion
...

## 6. What I would do next
...

## 7. References
- PMPP 4e §5.4 (tiling)
- PMPP 4e §6 (performance)
- CUTLASS docs, "Efficient GEMM in CUDA"
- `.cursor/skills/curriculum-plan/month-1-foundations.md`
```
