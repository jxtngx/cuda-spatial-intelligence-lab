# Workflow — Driving Cursor Through the Curriculum
<!-- subagent count: 12 (curriculum-mentor, cuda-tutor, cpp20-tutor,
     dgx-spark-engineer, cuda-perf-profiler, spatial-intel-researcher,
     model-deployer, langchain-deepagents-architect, cuda-code-reviewer,
     excalidraw-visualizer, manim-tutor, nemo-engineer) -->


How to actually use the commands, agents, and skills in this repo to
get through the 16 weeks. Read this once after
[`GETTING-STARTED.md`](./GETTING-STARTED.md) and before `/start-week 1`.
It pairs with [`SYLLABUS.md`](./SYLLABUS.md) (the *what* of the course)
and [`READING-GUIDE.md`](./READING-GUIDE.md) (the *how-to-read*).

> The TL;DR: **You drive the main thread. Slash commands are scripted
> workflows. Subagents are specialists you delegate to (cheaply,
> aggressively). Skills are reusable playbooks the agents and commands
> read.** Most of your day is `/start-week`, code, `/review-cuda`,
> `/profile-kernel`, `/lab-report`, `/checkpoint`. The rest is
> targeted agent calls when you're stuck.

---

## 1. Mental model — five layers, one workflow

Five things in this repo collaborate to teach you. Know what each is
*for* before you start invoking them.

| Layer | Lives in | Role | Examples |
|---|---|---|---|
| **You** | the chair | Form intent, judge output, write the lab. | "I want to implement tiled GEMM and beat 70% of cuBLAS." |
| **Main thread** | Cursor's chat | Orchestrates. Keeps you, the codebase, and the agents in sync. Reads files, writes code. | The conversation you're having right now. |
| **Slash commands** | `.cursor/commands/*.md` | Scripted, opinionated workflows. Each delegates to the right subagent and prints the result. | `/start-week`, `/review-cuda`, `/profile-kernel` |
| **Subagents** | `.cursor/agents/*.md` | Specialists with isolated context and a focused system prompt. | `cuda-tutor`, `cuda-perf-profiler`, `model-deployer` |
| **Skills** | `.cursor/skills/*/SKILL.md` | Reusable, hand-authored playbooks (templates, checklists). Agents read these. | `nsight-profiling`, `lab-notebook`, `weekly-checkpoint` |

The arrows go: **you → main thread → command → subagent → skill →
back through the chain.** Commands are the most common entry point;
direct agent calls are for off-script questions.

### Why use subagents at all?

Three reasons, in priority order:

1. **Context isolation.** Profiling a kernel pollutes your main
   thread with megabytes of `ncu` output. A subagent reads it, hands
   you a one-paragraph verdict, and the noise stays in its
   sandbox.
2. **Specialization.** `spatial-intel-researcher` knows Hartley's
   notation; `cpp20-tutor` knows the Iglberger Strategy/Visitor
   chapters. Each agent has a system prompt longer and more focused
   than anything you'd type into a chat.
3. **Reusability.** When the agent is good, every session is good.
   You're not re-bootstrapping the prompt every Monday.

If you remember nothing else: **delegate aggressively.** The main
thread should orchestrate, not drown in register-pressure analysis.

---

## 2. The seven slash commands — your daily driver

All commands live in `.cursor/commands/` and you invoke them with `/`
in the Cursor chat. Each is documented inline with a one-line
description; this section is the *when*-and-*how*.

### 2.1 `/start-week N`

**What it does.** Hands the week to `curriculum-mentor`, which reads
`.cursor/skills/curriculum-plan/month-{ceil(N/4)}-*.md`, finds Week N,
and prints the theme, learning objectives, required reading,
deliverable, performance target, 5-axis rubric, and which subagents
you should expect to need. Then scaffolds `labs/week-NN-<slug>/` from
`labs/_template/` if it doesn't exist yet.

**When.** Monday morning of every week, exactly once.

**How.**
```
/start-week 5
```

**What you should do with the output.**
1. Open `LAB.md` in the new lab folder. It's pre-populated with the
   spec — read it before you read anything else.
2. Open the chapters the mentor cited, *in the order it cited them*.
3. Don't open the editor yet. Reading first, code second.

**Pitfall.** Don't run `/start-week` again to "remind yourself".
Re-read the lab's `LAB.md` instead. Re-running may regenerate stale
state.

---

### 2.2 `/review-cuda`

**What it does.** Runs `git diff --stat HEAD` to see what changed,
then hands the diff to `cuda-code-reviewer`. The reviewer applies its
standard checklist (Correctness / Performance / Idiom / Lab-specific
rigor) and returns a structured output with `file:line` citations and
a verdict: `PASS | REWORK | BLOCKED`.

**When.** After every "I think this works" moment — a kernel
compiles and tests pass, or you've finished a refactor. Cheap; run it
often.

**How.**
```
/review-cuda
```

**What you should do with the output.**
- `PASS` → run `/profile-kernel` next.
- `REWORK` → fix the highest-leverage item, then re-run. Don't try to
  fix everything in one pass.
- `BLOCKED` → escalate to the named agent (the reviewer will tell you
  whether `cuda-tutor`, `cpp20-tutor`, or `cuda-perf-profiler` is
  right).

**Pitfall.** This is *static* review. It catches lifetime bugs,
modern-C++ idiom drift, kernel anti-patterns. It does *not* catch
"this kernel is slow because the L2 hit rate is 30%". That's
`/profile-kernel`.

---

### 2.3 `/profile-kernel`

**What it does.** Identifies the bench binary (asks if multiple),
confirms it built recently, then delegates to `cuda-perf-profiler` to
run the standard 4-step Nsight workflow:

1. `compute-sanitizer` (memcheck + racecheck + synccheck)
2. `nsys profile` with NVTX + CUDA traces → saved to `report/`
3. `ncu` with the standard section set + `--import-source on` →
   `report/ncu_<name>_v<N>.ncu-rep`
4. Read the report in canonical order, return the standard template:
   bottleneck → hypothesis → suggested change → expected effect.

The active lab's `LAB.md` performance target is the bar.

**When.** After `/review-cuda` returns `PASS`. Once per kernel
version. Always commit the `.ncu-rep` and `.qdrep` (or `.nsys-rep`)
to `report/`.

**How.**
```
/profile-kernel
```

**What you should do with the output.**
- The "suggested change" is your next task. Implement it; test;
  re-profile. That's one tuning loop.
- Tuning loops typically converge in 3–6 iterations. If you're past
  10, escalate to `cuda-tutor` ("am I optimizing the wrong thing?").

**Pitfall.** Don't profile broken code. `compute-sanitizer` is gate
1 for a reason.

---

### 2.4 `/lab-report`

**What it does.** Reads the `lab-notebook` skill template, inspects
the current lab folder (`src/`, `bench/`, `tests/`, the `*.ncu-rep`
files in `report/`), and authors or updates `report/LAB.md` with all
required sections: Spec, Hypothesis, Method per kernel version,
Results table, Discussion, Next steps, References. Cites primary
sources by section (`PMPP 4e §X.Y`, `Iglberger Ch N`, etc., not vague
"see PMPP").

**When.** End of every week, before `/checkpoint`. Or when you make a
substantive new finding mid-week and want it captured.

**How.**
```
/lab-report
```

**What you should do with the output.**
1. Read it. The agent gets ~80% right; the last 20% is your judgment.
2. Edit. The Discussion section is *your* voice — sharpen it.
3. Commit `report/LAB.md` alongside the `.ncu-rep` files.

**Pitfall.** Don't let the agent invent numbers. If the Results table
has a row but you don't have profile evidence, delete the row or run
the missing benchmark first.

---

### 2.5 `/checkpoint`

**What it does.** Hands the lab to `curriculum-mentor`, which grades
it against the 5-axis rubric in
`.cursor/skills/weekly-checkpoint/SKILL.md`. Each axis scored 0–4
with one specific `file:line` piece of evidence. Returns
`READY_TO_ADVANCE | REWORK | BLOCKED` with thresholds:

- Regular weeks: **14/20** to advance.
- Month boundaries (weeks 4, 8, 12): **17/20**.
- Week 16 (capstone): **18/20**.

**When.** Friday or Saturday of every week. Always after
`/lab-report`. Always *before* you mentally check out for the
weekend.

**How.**
```
/checkpoint
```

**What you should do with the output.**
- `READY_TO_ADVANCE` → next Monday is `/start-week N+1`.
- `REWORK` → the mentor names 1–3 highest-leverage fixes. Do those,
  re-run `/checkpoint`. Don't advance with debt.
- `BLOCKED` → tell the mentor what's in the way. The mentor can
  re-scope the lab.

**Pitfall.** Don't lobby the mentor. If it scored Performance 2/4
and you think it should be 3, the right move is to make the kernel
faster, not to argue.

---

### 2.6 `/research-paper <topic>`

**What it does.** Hands the topic to `spatial-intel-researcher`, which
returns a 1–2 page synthesis: classical framing (Hartley/Szeliski/
Torralba §) + modern framing (3–5 papers with arXiv IDs) + which
week of the curriculum it lands in + one open question. Saves to
`docs/research/<topic-slug>.md` so your personal literature index
accumulates.

**When.**
- Week before you'll need it (e.g. `Gaussian splatting` in Week 11).
- When a paper drops (e.g. a new Cosmos release).
- As a side quest when a concept in another lab fascinates you.

**How.**
```
/research-paper Gaussian splatting
/research-paper FlashAttention-3
/research-paper Cosmos-Reason fine-tuning
```

**What you should do with the output.**
- Skim the cited papers. Pick one to actively read; queue the others.
- The "open question" paragraph is a project starter. Save it.

**Pitfall.** Don't `/research-paper` everything. Five well-read
syntheses beat thirty browsed ones.

---

### 2.7 `/deploy-target {spark|sagemaker|bedrock}`

**What it does.** Identifies the model artifact in scope (current
week's `report/` or a path you give), then delegates to
`model-deployer` with the workflow from
`.cursor/skills/sagemaker-bedrock-deploy/SKILL.md` (and the Triton
workflow from the agent's system prompt for `spark`). Returns build
commands → resulting image / engine path → deploy command → smoke-test
snippet. Will *reject* the request and recommend an alternative if
the target is wrong (e.g. Bedrock for a diffusion model →
SageMaker BYOC).

**When.** Months 3-4. After a model fine-tune passes eval. Before
the dual-deploy bench in `bench/dual_deploy_bench.py`.

**How.**
```
/deploy-target spark
/deploy-target sagemaker
/deploy-target bedrock
```

**What you should do with the output.**
- Run the smoke test. If it passes, run the dual-deploy bench to get
  real cost/latency numbers vs the other target.
- Commit the engine / image tag in `deploy/<target>/`.

**Pitfall.** Don't deploy a model that hasn't passed an eval. The
deployer assumes the artifact is good.

---

### 2.8 Slash-command discipline

A few rules that compound over 16 weeks:

- **Run them in order.** `/start-week → code → /review-cuda →
  /profile-kernel → /lab-report → /checkpoint`. The order encodes the
  scientific method (hypothesis → implement → review → measure →
  write → grade).
- **One command per intent.** Don't pipeline `/review-cuda` and
  `/profile-kernel` in one message; you want to read each verdict.
- **Re-read, don't re-run.** Most "wait, what did the mentor say?"
  moments are answered by scrolling up, not re-invoking.
- **Trust the verdicts.** If `/checkpoint` says `REWORK`, the lab
  isn't done.

---

## 3. The twelve subagents — when each one shines

Slash commands cover ~80% of your interactions. The other 20% is
direct agent calls when you have an off-script question. Reach for an
agent by saying so explicitly:

> "Use `cuda-tutor` to explain why my GEMM kernel is bandwidth-bound
> when occupancy is 0.83 and arithmetic intensity is 32 FLOPs/byte."

### 3.1 Curriculum and pedagogy

| Agent | Use when |
|---|---|
| **`curriculum-mentor`** | You want the plan re-explained, the rubric clarified, or to negotiate an off-script lab. Owns `month-{1..4}-*.md`. *You almost always reach this one via `/start-week` or `/checkpoint`, not directly.* |

### 3.2 Language and CUDA tutors

| Agent | Use when |
|---|---|
| **`cuda-tutor`** | "Why is this kernel slow?" "What does this PTX line mean?" "Should I use cooperative groups here?" PMPP-aligned, sm_121-aware. |
| **`cpp20-tutor`** | "Is this the right place for `std::expected`?" "Type-erasure or virtual?" "Why won't this concept compile?" Iglberger / Stroustrup-aligned. |

### 3.3 Hardware and performance

| Agent | Use when |
|---|---|
| **`dgx-spark-engineer`** | Driver/toolkit/NGC issues. Unified-memory tuning. ConnectX-7 between two Sparks. "Which container tag should I be using?" |
| **`cuda-perf-profiler`** | Already invoked by `/profile-kernel`, but call directly when you want a roofline analysis, an occupancy-vs-tile-size sweep, or a SASS walkthrough. |

### 3.4 Research and CV

| Agent | Use when |
|---|---|
| **`spatial-intel-researcher`** | "Where in Hartley does the 8-point algorithm live?" "What's the SOTA for monocular depth on indoor scenes?" "Compare Gaussian splatting variants." |

### 3.5 Training, eval, and production

| Agent | Use when |
|---|---|
| **`nemo-engineer`** | Fine-tuning any HuggingFace causal LM / VLM (NeMo AutoModel), evaluating on standard benchmarks (`ns eval` over MATH/GPQA/IFEval/RULER/MMAU/etc.), generating synthetic data (NeMo Skills SDG), or scaling any of those workflows from the Spark to a Slurm cluster. The default path for Months 3-4 model work. |
| **`model-deployer`** | Already invoked by `/deploy-target`. Call directly for "should I quantize to FP8 before TRT or after?", "what's the right SageMaker instance type?", or "package this NeMo checkpoint for Triton". |
| **`langchain-deepagents-architect`** | Month 4 only. NextJS + DeepAgents structure, sensor I/O, agent tool design, RTSP/RealSense ingest, memory backend choice. |

### 3.6 Review

| Agent | Use when |
|---|---|
| **`cuda-code-reviewer`** | Already invoked by `/review-cuda`. Call directly when you want a focused review of one file or one PR. |

### 3.7 Visualization

| Agent | Use when |
|---|---|
| **`excalidraw-visualizer`** | You're about to write a wall of nouns describing a system. Asks you for a layout, then writes a `.excalidraw` file in `docs/` or `labs/.../report/`. |
| **`manim-tutor`** | A math/algorithm concept needs to *move* — multi-view geometry, parallel reduction, tiled GEMM, attention. Outputs runnable scenes in `docs/anim/`. |

### 3.8 The escalation tree

When something goes wrong, escalate in this order:

```
build / driver / NGC issue              → dgx-spark-engineer
CUDA runtime error                      → cuda-tutor
Host-side template / linker error       → cpp20-tutor
Kernel "works but slow"                 → cuda-perf-profiler  (or /profile-kernel)
Numerical wrong / lifetime bug          → cuda-code-reviewer  (or /review-cuda)
Conceptual confusion (paper / book)     → spatial-intel-researcher  (CV)
                                          cuda-tutor                (CUDA)
                                          cpp20-tutor               (C++)
Fine-tuning / SDG / benchmark eval      → nemo-engineer
TRT/Triton/SageMaker/Bedrock packaging  → model-deployer  (or /deploy-target)
"Am I even working on the right thing?" → curriculum-mentor
```

Pin this in your head. Half the time saved over 16 weeks is in
*choosing the right agent on the first try*.

---

## 4. Skills — the playbooks the agents read

You usually don't invoke a skill yourself; commands and agents read
them. But knowing they exist lets you say "use the
`nsight-profiling` skill" and skip a paragraph of explanation.

| Skill | Read by | Contains |
|---|---|---|
| `curriculum-plan/` | `curriculum-mentor`, `/start-week`, `/checkpoint` | The 16-week plan, week-by-week. |
| `weekly-checkpoint/` | `curriculum-mentor`, `/checkpoint` | The 5-axis rubric and threshold logic. |
| `lab-notebook/` | `/lab-report`, you | The `report/LAB.md` template. |
| `cuda-kernel-authoring/` | `cuda-tutor`, `cuda-code-reviewer` | Kernel design checklist (coalesced loads, occupancy, async copies, etc.). |
| `cpp20-modern-idioms/` | `cpp20-tutor`, `cuda-code-reviewer` | RAII, concepts, `std::span`, `std::expected`-style errors. |
| `nsight-profiling/` | `cuda-perf-profiler`, `/profile-kernel` | The 4-step `compute-sanitizer → nsys → ncu → read` workflow. |
| `dgx-spark-setup/` | `dgx-spark-engineer` | NGC tags, driver pinning, common Spark gotchas. |
| `cosmos-models/` | `spatial-intel-researcher`, `model-deployer` | Cosmos Predict/Reason/Transfer model facts. |
| `sagemaker-bedrock-deploy/` | `model-deployer`, `/deploy-target` | AWS deploy recipes. |
| `langchain-deepagents-nextjs/` | `langchain-deepagents-architect` | Month-4 app architecture. |

When you're writing your own prompt to an agent, you can say:

> "Use the `nsight-profiling` skill's section on memory throughput to
> diagnose this kernel."

That tells the agent which playbook to apply, and it skips
re-deriving the workflow from scratch.

---

## 5. The weekly rhythm — putting it together

A regular week (not a month-boundary or capstone). 20 hours, 5 days.

### Monday — orient

| | |
|---|---|
| **3 h reading** | The chapters `/start-week` lists. Bias toward PMPP / Iglberger early; Hartley / Torralba in Months 3-4. |
| **`/start-week N`** | Once. Read the output carefully. |
| **Open `LAB.md`** | Read the spec, the rubric, the perf target. |
| **First sketch** | Plain English in `LAB.md` Method section: *what* you'll implement, *why* it should hit the target. |

### Tuesday — first pass

| | |
|---|---|
| **Implement v1** | The naive correct version. Don't optimize yet. |
| **`cmake --build build && ctest`** | Make sure tests exist and pass. |
| **`/review-cuda`** | Catch idiom drift early. |
| **Direct agent call** | Whenever you hit a wall, ask `cuda-tutor` or `cpp20-tutor` *with the file open*. |

### Wednesday — measure

| | |
|---|---|
| **`/profile-kernel`** | Get baseline numbers. |
| **Read the Nsight report yourself** | Even though the agent summarized it. You're learning to read these. |
| **Plan v2** | Write the change in `LAB.md` Method §V2 *before* you code it. |

### Thursday — tune

| | |
|---|---|
| **Implement v2 (and v3 if needed)** | Iterate. |
| **`/review-cuda` then `/profile-kernel` between versions** | Don't skip review. |
| **Ask `cuda-perf-profiler` directly** | "Compare ncu reports v1 and v2 — what changed at the SASS level?" |

### Friday — write and grade

| | |
|---|---|
| **`/lab-report`** | Generate the report. |
| **Edit the Discussion section yourself** | The agent can't write your insights. |
| **Commit everything** | Source, tests, `.ncu-rep`, `LAB.md`. |
| **`/checkpoint`** | The gate. |
| **If `READY_TO_ADVANCE`** | Stop. Take the weekend. |
| **If `REWORK`** | Saturday morning, do the named fix, re-run. |

---

## 6. Common workflows — the four cycles

Every week is a permutation of these four loops. Memorize them.

### 6.1 The kernel cycle

The atomic unit of CUDA work.

```
write → ctest → /review-cuda → /profile-kernel
                     ↓                ↓
                  REWORK         suggested change
                     ↓                ↓
                  fix +           implement vN+1
                  re-run             ↓
                                  re-profile
```

You'll do this 3–6 times per week for the lead kernel.

### 6.2 The research cycle

Used Mondays and ad-hoc.

```
/research-paper <topic>
       ↓
read top-3 cited papers
       ↓
identify the one technique to try
       ↓
update LAB.md Method with hypothesis
       ↓
back to the kernel cycle
```

### 6.3 The deploy cycle (Months 3-4)

```
fine-tune model          (nemo-engineer: AutoModel / Framework)
       ↓
gate with eval           (nemo-engineer: ns eval over chosen benchmarks)
       ↓
quantize / export        (model-deployer; PTQ/QAT via NeMo Model Optimizer)
       ↓
/deploy-target spark     → smoke test → bench
/deploy-target sagemaker → smoke test → bench
       ↓
compare cost / latency in bench/dual_deploy_bench.py
       ↓
choose target, document choice in LAB.md
```

### 6.4 The debug cycle

```
test fails OR compute-sanitizer flags something
       ↓
escalate using §3.8 tree
       ↓
agent suggests minimal repro
       ↓
implement repro → run → fix
       ↓
re-run /review-cuda + the failed test
       ↓
write a one-line "what I learned" in LAB.md Discussion
```

That last line is the part most people skip. Don't.

---

## 7. Patterns that work

A few habits that compound:

- **Open the file you're asking about.** Subagents see your open
  files. "Why is this slow?" with the kernel open is a different
  question from "Why is CUDA slow?" with nothing open.
- **Quote `file:line` when escalating.** "`labs/week-05-reduce/src/
  reduce.cu:42` — why does this race?" routes you to the answer
  faster than narrative.
- **Demand citations.** When `cuda-tutor` says "this is a memory-bound
  kernel", ask "cite the section". The good agents already do this;
  reinforce the habit.
- **Capture insights in `LAB.md` immediately.** If you don't write it
  down within an hour, you'll re-derive it next month.
- **Diagram before you re-architect.** When a system has more than
  three moving pieces, ask `excalidraw-visualizer` first. A 5-minute
  diagram saves a 2-day refactor.
- **Animate before you teach.** When a math concept won't stick, ask
  `manim-tutor` for a 30-second scene. Then watch your own scene three
  times. The third time it'll click.
- **Use `/research-paper` as bookmarking.** It writes to
  `docs/research/`, which becomes your personal literature index by
  Month 4.

## 8. Antipatterns to avoid

- **Asking the main thread to do specialist work.** "Tell me about
  Blackwell tensor cores" wastes context that a subagent would
  isolate. Say "use `cuda-tutor` to..." instead.
- **Skipping `/review-cuda` because "it works".** Static review
  catches lifetime bugs and idiom drift that profiles will *never*
  catch. Run it every time.
- **Skipping `/profile-kernel` because "it's fast enough".** Fast
  enough for what? The lab has a target. Profile and *prove* you hit
  it.
- **Never running `/checkpoint`.** Without the grade, the curriculum
  is just a reading list. The verdict is the gate.
- **Letting `/lab-report` generate the Discussion section unedited.**
  The agent can summarize numbers. It cannot synthesize what you
  learned. That's the whole point of the rubric's Writeup axis.
- **Re-running a command to "see if it changes its mind".** It
  shouldn't. If it does, the input changed; document why.
- **Working without an open editor.** Subagents work better when they
  can see your code. Use Cursor's split panes; keep `LAB.md`, the
  active source file, and the chat side-by-side.

---

## 9. Power tips

A handful of moves that separate a smooth week from a rough one:

- **Ask the agent to ask you.** When you don't know what to specify,
  say: "Ask me three questions before you start." The good agents
  oblige.
- **Force a hand-off explicitly.** "After you finish this kernel
  review, hand off to `cuda-perf-profiler` to plan the next
  benchmark." Saves a round trip.
- **Pin the rubric in chat.** Paste the 5-axis rubric (or open
  `.cursor/skills/weekly-checkpoint/SKILL.md`) before substantive
  work. The main thread will start applying it without prompting.
- **Use Plan mode for architecture decisions.** Before any
  multi-file refactor, switch Cursor to Plan mode (top-right). The
  thread becomes read-only and discussion-first.
- **Generate a diagram alongside any new lab folder.** The first day
  of every lab, ask `excalidraw-visualizer` to sketch the dataflow.
  It becomes the cover image for `report/LAB.md`.
- **Run two profiles, not one.** `/profile-kernel` on `vN`, then on
  `vN+1`. Then ask `cuda-perf-profiler` to *diff* them. The diff is
  where the lesson lives.
- **End every week with a one-line `what I'd do differently`.** Pin
  it at the top of `LAB.md` Discussion. By Week 16 you'll have 16
  lines that compress more wisdom than any blog post.

---

## 10. Where to look when this guide isn't enough

| Question | Read |
|---|---|
| What chapters this week? | `LAB.md` after `/start-week`, or the relevant `month-N-*.md`. |
| How do I structure `report/LAB.md`? | `.cursor/skills/lab-notebook/SKILL.md` |
| What's the rubric? | `.cursor/skills/weekly-checkpoint/SKILL.md` |
| Which Nsight section means what? | `.cursor/skills/nsight-profiling/SKILL.md` |
| Why is my driver / NGC tag wrong? | [`GETTING-STARTED.md`](./GETTING-STARTED.md) §1, §5 + `dgx-spark-engineer` |
| When should I use which book? | [`READING-GUIDE.md`](./READING-GUIDE.md) |
| What's the 4-month arc? | [`SYLLABUS.md`](./SYLLABUS.md) §4-§5 + `SYLLABUS.excalidraw` |
| What does each agent know? | The agent's own `.cursor/agents/<name>.md` — they're short. |

When in doubt, **open the agent file and read it**. The system prompts
are public, and they'll tell you exactly what the agent will and
won't do.

---

Now go run `/start-week 1`.
