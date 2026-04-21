# Lab NN — <Title>

> Replace this template once `/start-lab NN` has been run. The
> `curriculum-mentor` populates this file from
> `.cursor/skills/curriculum-plan/month-X-*.md`.

## 0. Intro

*One paragraph in plain English: what is this lab? Why does it exist
in the curriculum arc? What will the student have built by the end?*

> **New terms this lab.** See [`GLOSSARY.md`](./GLOSSARY.md) in this
> folder. Read it before §3 Spec — it covers the CUDA, C++20, CV /
> Spatial Intelligence (and from Lab 02 on, Python bindings) terms
> that are introduced for the first time in this lab.

## Plan of work — order of operations

*Required section. The `curriculum-mentor` must populate every
checkbox below with lab-specific text when scaffolding. Phase
letters are referenced by `/checkpoint` when grading. Work
top-to-bottom; do not move past a checkbox until it's green.*

### A. Read first (do not skip)

- [ ] TODO primary reading 1 (chapter / section)
- [ ] TODO primary reading 2
- [ ] Skim this lab's [`GLOSSARY.md`](./GLOSSARY.md).

### B. Bring the scaffold up on Spark

- [ ] Toolchain check (CUDA 13, CMake ≥ 3.28, Ninja, sm_121, plus
      any lab-specific deps — cuBLAS / CUB / CUTLASS / torch / …).
- [ ] `cmake -S . -B build -G Ninja && cmake --build build -j`.
- [ ] Baseline `ctest --test-dir build --output-on-failure`. Note
      which cases pass at scaffold time and which require Phase D.

### C. Write §5 Hypothesis *before* you optimize

- [ ] State the predicted bottleneck for each version and the
      Nsight counter that will confirm it. Predict the perf number
      as a fraction of the §3 baseline. Commit to §5 *before* you
      measure.

### D. Implement the §4 TODOs and turn ctest green

- [ ] TODO file-specific TODOs from §4 Your task
- [ ] `ctest --test-dir build --output-on-failure` — green at all
      sizes for all versions.

### E. Get the Python bindings green   *(Lab 02+ only — delete in Lab 01)*

- [ ] `pytest python/` — numerics vs the torch reference and the
      wrapper-overhead bound (Tier-A coarse rule in Labs 2-4;
      strict 5%-of-kernel-time rule from Lab 05 on).

### F. Run the bench and hit the perf target

- [ ] `./build/bench_<name>` on Spark, capture into the §7 table.
- [ ] **Target:** TODO restate the §3 perf target.
- [ ] If you miss it, iterate and document each attempt in §8.

### G. Profile (evidence for the perf claim)

Follow `.cursor/skills/nsight-profiling/SKILL.md`. Commit raw
artifacts under `report/`.

- [ ] `report/nsys_<name>.qdrep` — Nsight Systems trace.
- [ ] `report/ncu_<name>.ncu-rep` — Nsight Compute report with
      Speed of Light + Memory Workload Analysis at minimum.
- [ ] (Checkpoint labs only — 4 / 8 / 12 / 16) full sections, plus
      `report/roofline.png` if applicable.

### H. Write it up

- [ ] Fill §7 Results from the bench output (every row).
- [ ] §8 Discussion: cite the Nsight section + counter for each
      claim in §5 Hypothesis. Honesty about misses counts.
- [ ] §10 What I would do next.
- [ ] Run `/lab-report` to polish (`.cursor/skills/lab-notebook/SKILL.md`).

### I. Self-grade

- [ ] `/checkpoint` against the 5-axis rubric in
      `.cursor/skills/weekly-checkpoint/SKILL.md`.
      **≥ 14/20** to advance. **≥ 17/20** in checkpoint labs
      (4 / 8 / 12 / 16).

### Definition of done for Lab NN

`ctest` is green; `pytest python/` is green (Lab 02+); bench
shows TODO restate perf target; `report/` holds the named Nsight
artifacts; `LAB.md` §5, §7, §8, §10 are written.

## 1. What you will learn

By the end of this lab you will be able to:

### CUDA
- TODO outcome 1
- TODO outcome 2

### C++20
- TODO outcome 1

### CV / Spatial Intelligence
- TODO outcome 1 (or *N/A this week — no new CV/SI material*)

### Python bindings
*(Lab 02+ only — delete this subsection in Lab 01.)*
- TODO outcome 1

## 2. Prerequisites

**Readings (do these first):**
- PMPP 4e Ch X §Y.Z
- Iglberger Ch A
- (other primary sources)

**Prior labs' deliverables you need working:**
- Lab NN-1: <what>

**Toolchain checks:**
- `nvidia-smi`, `nvcc --version`, CMake ≥ 3.28 (see
  [`docs/GETTING-STARTED.md`](../../docs/GETTING-STARTED.md) if
  anything is missing).

## 3. Spec

*Inputs / outputs / dtypes / numerical tolerance / performance
target. This is the contract; §4 Your task tells you which parts of
the contract you implement.*

### Inputs / outputs / dtypes
TODO

### Numerical tolerance
TODO

### Performance target
TODO

## 4. Your task

*This is the section the student should read most carefully.* Lists
the files in `src/` (and `python/` from Lab 02 on) that are blank /
stubbed at this lab's scaffolding tier, and what "done" looks like.

This lab is **Tier <A | B | C | D>**. See
[`labs/_template/README.md`](../_template/README.md) for what that
means. Concretely:

- `src/<file>.<cu|cpp|hpp>` — TODO description of what's blank and
  what to write.
- `tests/<file>.cpp` — TODO (provided / you-write).
- `bench/<file>.cpp` — TODO (provided / you-write).
- `python/<lab>_ext.py` — *(Lab 02+ only)* TODO description.
- `python/test_<lab>.py` — *(Lab 02+ only)* TODO description.

**Definition of done.** All `ctest` targets pass, `pytest python/`
passes (Lab 02+), bench reports a number that meets the §3 target,
and `report/` contains a Nsight artifact when the rubric requires
one.

## 5. Hypothesis

*Write this BEFORE you start coding the optimized version.* What is
the bottleneck you predict? What will Nsight show? What perf number
do you expect?

TODO

## 6. Method

*Versions you'll implement, in order. Each is a short paragraph.*

### v0 — naive
TODO

### v1 — TODO
TODO

## 7. Results

| Version | Time (ms) | GFLOP/s or GB/s | % of baseline |
|---|---|---|---|
| v0 |  |  |  |
| v1 |  |  |  |

Reference Nsight reports:
- `report/nsys_<name>.qdrep`
- `report/ncu_<name>.ncu-rep`

## 8. Discussion

Did the predicted bottleneck show up? What surprised you in the
profile? What would you change in your hypothesis next time?

TODO

## 9. References

- PMPP 4e §
- Iglberger Ch
- Hartley §
- CUDA C++ Programming Guide §
- Curriculum: `.cursor/skills/curriculum-plan/month-X-*.md`
- Skills: `.cursor/skills/cuda-kernel-authoring/SKILL.md`,
  `.cursor/skills/python-bindings/SKILL.md` *(Lab 02+)*

## 10. What I would do next

*A stretch prompt, not a substitute for §4. One paragraph: what's the
next optimization or extension you'd attempt if you had another day?*

TODO
