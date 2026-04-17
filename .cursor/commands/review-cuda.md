---
description: Review the current CUDA / C++20 changes for correctness, performance, idiom, and lab rigor. Delegates to cuda-code-reviewer.
---

The user invoked `/review-cuda`. Run a structured review of the current
CUDA + C++ work.

Workflow:

1. Run `git diff --stat HEAD` to see what changed (and `git diff HEAD`
   if the user asks for full diff context).
2. Delegate to the **`cuda-code-reviewer`** subagent with:
   > "Review the current changes (uncommitted + last commit) in the
   > active lab folder. Use the cuda-code-reviewer checklist
   > (Correctness / Performance / Idiom / Lab-specific rigor) and
   > produce the standard structured output with file:line citations
   > and a verdict (PASS | REWORK | BLOCKED)."
3. Print the reviewer's output verbatim.
4. If verdict is REWORK or BLOCKED, suggest the single highest-leverage
   fix the user should make next, and recommend the subagent to consult
   (`cuda-tutor`, `cpp20-tutor`, or `cuda-perf-profiler`).
