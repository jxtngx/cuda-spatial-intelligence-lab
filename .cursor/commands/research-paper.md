---
description: Synthesize a short literature review on a Spatial Intelligence / CUDA topic, citing primary sources with arXiv IDs and pointing back to the relevant week of the curriculum.
---

The user invoked `/research-paper` with a topic argument (e.g.
"Gaussian splatting", "FlashAttention-3", "Cosmos-Reason fine-tuning").
If the topic is missing, ask once.

Workflow:

1. Delegate to **`spatial-intel-researcher`** with:
   > "Produce a 1-2 page synthesis on `<topic>`:
   > - Frame the problem in classical (Hartley/Szeliski/Torralba §) and
   >   modern (paper) terms.
   > - 3-5 primary references with arXiv IDs (or DOIs / book §).
   > - One paragraph on how this lands in the curriculum: which week,
   >   which lab, which subagent to consult.
   > - One paragraph naming a non-obvious open question."
2. Print the synthesis verbatim.
3. Save it to `docs/research/<topic-slug>.md` (create folder if needed)
   so it accumulates as a personal literature index.
