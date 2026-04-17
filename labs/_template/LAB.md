# Week NN — <Title>

> Replace this template once `/start-week NN` has been run. The
> `curriculum-mentor` populates this file from
> `.cursor/skills/curriculum-plan/month-X-*.md`.

## 0. Intro

*One paragraph in plain English: what is this lab? Why does it exist
in the curriculum arc? What will the student have built by the end?*

> **New terms this week.** See [`GLOSSARY.md`](./GLOSSARY.md) in this
> folder. Read it before §3 Spec — it covers the CUDA, C++20, CV /
> Spatial Intelligence (and from Week 2 on, Python bindings) terms
> that are introduced for the first time this week.

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
*(Week 2+ only — delete this subsection in Week 1.)*
- TODO outcome 1

## 2. Prerequisites

**Readings (do these first):**
- PMPP 4e Ch X §Y.Z
- Iglberger Ch A
- (other primary sources)

**Prior weeks' deliverables you need working:**
- Week NN-1: <what>

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
the files in `src/` (and `python/` from Week 2 on) that are blank /
stubbed at this week's scaffolding tier, and what "done" looks like.

This week is **Tier <A | B | C | D>**. See
[`labs/_template/README.md`](../_template/README.md) for what that
means. Concretely:

- `src/<file>.<cu|cpp|hpp>` — TODO description of what's blank and
  what to write.
- `tests/<file>.cpp` — TODO (provided / you-write).
- `bench/<file>.cpp` — TODO (provided / you-write).
- `python/<lab>_ext.py` — *(Week 2+ only)* TODO description.
- `python/test_<lab>.py` — *(Week 2+ only)* TODO description.

**Definition of done.** All `ctest` targets pass, `pytest python/`
passes (Week 2+), bench reports a number that meets the §3 target,
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
  `.cursor/skills/python-bindings/SKILL.md` *(Week 2+)*

## 10. What I would do next

*A stretch prompt, not a substitute for §4. One paragraph: what's the
next optimization or extension you'd attempt if you had another day?*

TODO
