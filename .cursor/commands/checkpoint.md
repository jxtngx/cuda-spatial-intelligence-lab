---
description: End-of-week checkpoint. Grades the active lab against the 5-axis rubric and returns READY_TO_ADVANCE | REWORK | BLOCKED with concrete next steps. Delegates to curriculum-mentor.
---

The user invoked `/checkpoint`. Grade the current week's lab against the
weekly rubric.

Workflow:

1. Determine the current week from the active lab folder name. If
   ambiguous, ask.
2. Delegate to the **`curriculum-mentor`** subagent with:
   > "Grade the user's Week {N} lab in `labs/week-{NN}-<slug>/` against
   > the 5-axis rubric defined in
   > `.cursor/skills/weekly-checkpoint/SKILL.md`. Use the threshold
   > appropriate for this week (regular = 14/20; weeks 4/8/12 = 17/20;
   > week 16 = 18/20). For each axis, score 0-4 with one specific piece
   > of file:line evidence. Return the standard verdict template with
   > the highest-leverage 1-3 fixes if REWORK."
3. Print the mentor's output verbatim.
4. If verdict is `READY_TO_ADVANCE`, congratulate briefly and suggest
   `/start-week {N+1}`.
5. If `REWORK`, suggest the most appropriate subagent to consult for
   the top fix.
6. If `BLOCKED`, ask the user what is in the way and offer to
   triage with the mentor.
