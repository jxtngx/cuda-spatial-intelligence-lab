# `labs/_template/` — the lab scaffold

Every weekly lab is a copy of this folder, populated by the
`curriculum-mentor` subagent when you run `/start-week N`. This file
documents the **scaffolding tiers** that determine *how much code is
already written for you* in each lab.

The pedagogy is DataCamp-style:

> **show → fill in the blanks → from-scratch implementation.**

The progression happens by month, so each new domain (foundations,
advanced CUDA, spatial intelligence, production) gets its own ramp.

## The four tiers

| Tier | Months | Weeks | What `src/` looks like |
|---|---|---|---|
| **A** | Month 1 | 1-4 | **Show + small fill-in.** Compiling code with 2-4 `// TODO(student):` markers per file (inner-loop body, launch config, one assertion). Tests + bench + CMake fully provided. |
| **B** | Month 2 | 5-8 | **Function-body blanks.** Headers, signatures, types, RAII wrappers, and tests are provided. Function bodies are stubbed with `// TODO(student):` plus a contract-describing comment block. Bench skeleton provided; student fills the kernel call. |
| **C** | Month 3 | 9-12 | **Signatures + spec only.** Headers declare the API and the spec; `.cu` / `.cpp` files contain only `#include`s and the function signature with `throw std::logic_error("not implemented");`. Tests provided so the student knows what "done" means. |
| **D** | Month 4 | 13-16 | **Pure spec.** `src/` is empty (`.gitkeep`). Student designs the API from `LAB.md` §3 + §4 alone. CMake + tests scaffold provided so the build system isn't the fight; `tests/` may have TODO assertions encouraging the student to design their own contract. |

**The mentor enforces these tiers** when scaffolding
`labs/week-NN-*/`. Tier is computed as
`tier = ['A','B','C','D'][ceil(N/4) - 1]`.

## What's the same in every tier

Regardless of tier, every lab gets:

- `LAB.md` populated end-to-end (Intro, What you will learn,
  Prerequisites, Spec, Your task, Hypothesis stub, Method stub,
  Results table, Discussion stub, References, What I would do next).
- `GLOSSARY.md` populated with **only the new terms for this week**
  (CUDA, C++20, CV / Spatial Intelligence, and from Week 2 on,
  Python bindings). The mentor reads earlier weeks' glossaries first
  to avoid duplication.
- `CMakeLists.txt` that builds, even if `src/` is empty.
- `tests/` directory with at least a placeholder so `ctest` returns
  meaningfully.
- `bench/` directory (skeleton appropriate to the tier).
- `report/` directory for Nsight artifacts and `/lab-report` output.
- **Starting Week 2:** a `python/` directory with a
  `torch.utils.cpp_extension`-based wrapper (`<lab>_ext.py`) and a
  `pytest` (`test_<lab>.py`), both at the same tier as `src/`. See
  [`.cursor/skills/python-bindings/SKILL.md`](../../.cursor/skills/python-bindings/SKILL.md).

## Tier promotion / regression

The mentor will not soften a tier because the work is hard. If you're
stuck:

1. Re-read the prerequisites in `LAB.md` §2.
2. Use `/review-cuda` or invoke the relevant tutor subagent
   (`cuda-tutor`, `cpp20-tutor`, `spatial-intel-researcher`).
3. If genuinely blocked, run `/checkpoint` and ask the mentor to
   diagnose — *do not* ask for the tier to be lowered.

## Adding ad-hoc work outside the curriculum

Copy `_template/` directly:

```bash
cp -r labs/_template labs/scratch-<slug>
```

Scratch labs aren't graded against the rubric; the tier system is
informational there.
