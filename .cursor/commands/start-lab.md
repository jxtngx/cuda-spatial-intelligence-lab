---
description: Open a curriculum lab. Reads the relevant month file, prints objectives + readings + lab spec + rubric, and scaffolds labs/lab-NN-*/ at the correct tier if missing.
---

You are about to start a curriculum lab. The user invoked `/start-lab`
with an integer argument N (1-16). If N is missing, ask once.

The curriculum is **16 labs**, paced roughly one per week over four
months. Labs 4, 8, 12, 16 are checkpoint labs with a stricter rubric
(≥ 17/20, not 14).

Workflow:

1. Compute the **scaffolding tier** from N:

   ```
   tier = ['A', 'B', 'C', 'D'][ceil(N / 4) - 1]
   ```

   (Labs 1-4 = Tier A, 5-8 = B, 9-12 = C, 13-16 = D. See
   `labs/_template/README.md` for what each tier means.)

2. Delegate to the **`curriculum-mentor`** subagent. Give it the prompt:

   > "The user is starting Lab {N} (Tier {tier}). Read
   > `.cursor/skills/curriculum-plan/month-{ceil(N/4)}-*.md`, find Lab
   > {N} (titled "Week {N}" inside the curriculum-plan files for
   > historical reasons — that is the same lab), and produce the
   > standard `/start-lab` output as defined in your "Output template
   > for /start-lab N" section: Intro, Learning objectives split into
   > CUDA / C++20 / CV / (Lab 02+) Python bindings buckets, Required
   > reading, Lab deliverable, Your task scoped to Tier {tier},
   > Performance targets, 5-axis rubric, and recommended subagents.
   >
   > Then create `labs/lab-{NN}-<slug>/` if it does not yet exist,
   > copying from `labs/_template/`. Populate:
   >
   > - `LAB.md` — every section in the template populated end-to-end.
   >   Especially:
   >     * §0 Intro (plain English, one paragraph).
   >     * **`## Plan of work — order of operations`** — required.
   >       Phases A through I, every checkbox lab-specific (cite the
   >       actual readings, files, perf numbers, Nsight artifact
   >       names). Phase E (Python bindings) is omitted only for
   >       Lab 01.
   >     * §1 What you will learn (bucketed: CUDA / C++20 / CV-SI /
   >       Python bindings).
   >     * §4 Your task (file-by-file, scoped to Tier {tier}).
   >     * §5 Hypothesis stub that the student writes *before*
   >       optimizing.
   >     * §7 Results table with rows pre-populated for every version
   >       the lab will benchmark.
   > - `GLOSSARY.md` — new terms only. Read every existing
   >   `labs/lab-*/GLOSSARY.md` first to avoid duplication. Four
   >   buckets (CUDA, C++20, Spatial Intelligence / CV, and Lab 02+
   >   Python bindings). Empty buckets get an explicit
   >   `*No new terms introduced this lab.*`.
   > - `src/` — scaffolded at Tier {tier} (see your "Scaffolding
   >   tiers" section). Tier A = compiling code with 2-4 TODOs per
   >   file. Tier B = function-body blanks. Tier C = signatures only
   >   throwing `not implemented`. Tier D = empty (`.gitkeep`).
   > - `tests/`, `bench/`, `report/`, `CMakeLists.txt` per the
   >   template.
   > - **For N >= 2**: a `python/` directory containing
   >   `<lab>_ext.py` (torch.utils.cpp_extension JIT loader + smoke
   >   test) and `test_<lab>.py` (pytest), scaffolded at Tier
   >   {tier}. Cite `.cursor/skills/python-bindings/SKILL.md`."

3. After the mentor returns, print its output to the user verbatim.

4. Suggest the next concrete action ("Read the chapters first.
   Skim `GLOSSARY.md`. Then `/review-cuda` after your first compile.").

Notes:
- Month boundaries: labs 1-4 = month-1; labs 5-8 = month-2;
  labs 9-12 = month-3; labs 13-16 = month-4.
- The curriculum-plan files under `.cursor/skills/curriculum-plan/`
  still use "Week N" as the section heading. Treat "Week N" inside
  those files as synonymous with "Lab N" — same content, different
  framing. The student-facing convention is **labs**.
- Use the `curriculum-mentor` subagent rather than reading the month
  file directly in the main thread, so the main context stays clean.
- The mentor is responsible for tier enforcement, glossary
  de-duplication, and writing a complete Plan of work. Do not
  attempt any of the three in the main thread.
