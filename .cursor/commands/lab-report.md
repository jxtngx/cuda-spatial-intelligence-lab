---
description: Generate or update report/LAB.md for the current week's lab using the lab-notebook skill template, populated from current code, tests, and Nsight reports in report/.
---

The user invoked `/lab-report`. Produce a rigorous `report/LAB.md`
following the **`lab-notebook`** skill's template.

Workflow:

1. Determine the current lab folder. If unclear, list `labs/week-*` and
   ask the user to pick.
2. Read the `lab-notebook` skill (`.cursor/skills/lab-notebook/SKILL.md`)
   for the required structure and tone rules.
3. Inspect the lab folder:
   - `src/` and `bench/` for kernel versions.
   - `tests/` for what is tested.
   - `report/*.ncu-rep` and `report/*.qdrep` (or summary `.txt` files)
     for profile data.
   - Existing `report/LAB.md` (preserve what's good, sharpen what's not).
4. Author / update `report/LAB.md` with all required sections (Spec,
   Hypothesis, Method per version, Results table, Discussion, Next, References).
5. **Cite primary sources by section** (PMPP 4e §X.Y, Iglberger Ch N,
   Hartley §X.Y, paper title + arXiv). No vague citations.
6. After writing, print a one-line verdict on which axis of the rubric
   is now strongest and which is still weakest.

If you can't fill a section honestly (e.g. no Nsight report yet), leave
a `TODO:` comment in that section rather than fabricate numbers.
