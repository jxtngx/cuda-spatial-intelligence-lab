---
description: Open a curriculum week. Reads the relevant month file, prints objectives + readings + lab spec + rubric, and scaffolds labs/week-NN-*/ if missing.
---

You are about to start a curriculum week. The user invoked `/start-week`
with an integer argument N (1-16). If N is missing, ask once.

Workflow:

1. Delegate to the **`curriculum-mentor`** subagent. Give it the prompt:

   > "The user is starting Week {N}. Read
   > `.cursor/skills/curriculum-plan/month-{ceil(N/4)}-*.md`, find Week
   > {N}, and produce the standard `/start-week` output (theme, learning
   > objectives, required reading with chapters, lab deliverable, perf
   > target, 5-axis rubric, recommended subagents). Then create
   > `labs/week-{NN}-<slug>/` if it does not yet exist with the standard
   > skeleton (`LAB.md`, `CMakeLists.txt`, `src/`, `bench/`, `tests/`,
   > `report/`)."

2. After the mentor returns, print its output to the user verbatim.

3. Suggest the next concrete action ("Read the chapters first. Then
   `/review-cuda` after your first compile.").

Notes:
- Month boundaries: weeks 1-4 = month-1; weeks 5-8 = month-2;
  weeks 9-12 = month-3; weeks 13-16 = month-4.
- Use the `curriculum-mentor` subagent rather than reading the month
  file directly in the main thread, so the main context stays clean.
