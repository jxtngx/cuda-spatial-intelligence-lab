---
description: Open a curriculum week. Reads the relevant month file, prints objectives + readings + lab spec + rubric, and scaffolds labs/week-NN-*/ at the correct tier if missing.
---

You are about to start a curriculum week. The user invoked `/start-week`
with an integer argument N (1-16). If N is missing, ask once.

Workflow:

1. Compute the **scaffolding tier** from N:

   ```
   tier = ['A', 'B', 'C', 'D'][ceil(N / 4) - 1]
   ```

   (Weeks 1-4 = Tier A, 5-8 = B, 9-12 = C, 13-16 = D. See
   `labs/_template/README.md` for what each tier means.)

2. Delegate to the **`curriculum-mentor`** subagent. Give it the prompt:

   > "The user is starting Week {N} (Tier {tier}). Read
   > `.cursor/skills/curriculum-plan/month-{ceil(N/4)}-*.md`, find Week
   > {N}, and produce the standard `/start-week` output as defined in
   > your "Output template for /start-week N" section: Intro, Learning
   > objectives split into CUDA / C++20 / CV / (Week 2+) Python
   > bindings buckets, Required reading, Lab deliverable, Your task
   > scoped to Tier {tier}, Performance targets, 5-axis rubric, and
   > recommended subagents.
   >
   > Then create `labs/week-{NN}-<slug>/` if it does not yet exist,
   > copying from `labs/_template/`. Populate:
   >
   > - `LAB.md` — all 11 sections filled in end-to-end. Especially
   >   §0 Intro (plain English), §1 What you will learn (bucketed),
   >   §4 Your task (file-by-file, scoped to Tier {tier}).
   > - `GLOSSARY.md` — new terms only. Read every existing
   >   `labs/week-*/GLOSSARY.md` first to avoid duplication. Four
   >   buckets (CUDA, C++20, Spatial Intelligence / CV, and Week 2+
   >   Python bindings). Empty buckets get an explicit
   >   `*No new terms introduced this week.*`.
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
- Month boundaries: weeks 1-4 = month-1; weeks 5-8 = month-2;
  weeks 9-12 = month-3; weeks 13-16 = month-4.
- Use the `curriculum-mentor` subagent rather than reading the month
  file directly in the main thread, so the main context stays clean.
- The mentor is responsible for tier enforcement and glossary
  de-duplication; do not attempt either in the main thread.
