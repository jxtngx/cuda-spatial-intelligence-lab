---
description: End-of-lab checkpoint. Grades the active lab against the 5-axis rubric and returns READY_TO_ADVANCE | REWORK | BLOCKED with concrete next steps. Delegates to curriculum-mentor.
---

The user invoked `/checkpoint`. Grade the current lab against the
5-axis rubric.

Workflow:

1. Determine the current lab from the active `labs/lab-NN-<slug>/`
   folder name. If ambiguous, ask.
2. Delegate to the **`curriculum-mentor`** subagent with:
   > "Grade the user's Lab {N} in `labs/lab-{NN}-<slug>/` against
   > the 5-axis rubric defined in
   > `.cursor/skills/weekly-checkpoint/SKILL.md`. Use the threshold
   > appropriate for this lab (regular = 14/20; checkpoint labs
   > 4 / 8 / 12 = 17/20; lab 16 capstone = 18/20). For each axis,
   > score 0-4 with one specific piece of `file:line` evidence.
   > Reference the lab's `## Plan of work` phase letters (A-I) when
   > citing what is or isn't done. Return the standard verdict
   > template with the highest-leverage 1-3 fixes if REWORK."
3. Print the mentor's output verbatim.
4. If verdict is `READY_TO_ADVANCE`, congratulate briefly and suggest
   `/start-lab {N+1}`.
5. If `REWORK`, suggest the most appropriate subagent to consult for
   the top fix.
6. If `BLOCKED`, ask the user what is in the way and offer to
   triage with the mentor.
