# Syllabus — CUDA + C++20 + Spatial Intelligence (Special Topics)

A 4-month, 16-week, deep-water special-topics course taught **by Claude in Cursor** inside the `cuda-spatial-intelligence-lab` repo, on an NVIDIA DGX Spark.

> **Companion docs.**
> - [`GETTING-STARTED.md`](./GETTING-STARTED.md) — **read this first.**
>   Verify your driver, CUDA toolkit, host C++ compiler, CMake, and NGC container setup.
>   Includes `nvidia-smi` and `nvcc` cheat sheets.
> - [`WORKFLOW.md`](./WORKFLOW.md) — how to actually drive the slash commands and partner with the subagents.
>   The four cycles, the escalation tree, the weekly rhythm.
> - [`CURSOR-DOCS.md`](./CURSOR-DOCS.md) — the third-party docs to import into Cursor's `@Docs` index (PTX, C++20, NeMo, TensorRT-LLM, Cosmos, LangChain DeepAgents, …).
>   Import once, before Week 1.
> - [`SYLLABUS.excalidraw`](./SYLLABUS.excalidraw) — visual companion: the 4-month arc, the daily Cursor loop, and the subagent roster on one canvas.
>   Open in Cursor with the Excalidraw extension.
> - [`READING-GUIDE.md`](./READING-GUIDE.md) — how to actually use the six textbooks.
> - [`READING-GUIDE.excalidraw`](./READING-GUIDE.excalidraw) — visual companion: how the six books connect and the four reading modes.
> - [`../.cursor/skills/curriculum-plan/SKILL.md`](../.cursor/skills/curriculum-plan/SKILL.md) — the canonical week-by-week plan (this syllabus quotes from it).
> - [`../README.md`](../README.md) — project overview.
> - [`../AGENTS.md`](../AGENTS.md) — context the AI agents operate under.

---

## 1. Course identity

| Field | Value |
|---|---|
| Title | CUDA + C++20 + Spatial Intelligence |
| Code (informal) | SI-621 |
| Format | Self-paced, lab-driven, taught by AI subagents in Cursor |
| Duration | 16 weeks (~20 hrs/week of focused work) |
| Workload | 3 hrs/weekday + a long weekend session |
| Hardware | NVIDIA DGX Spark (Grace Blackwell, sm_121, 128 GB unified memory) |
| Prerequisites | Strong Python; solid PyTorch; comfort with linear algebra and gradient-based optimization; some C exposure helpful but not required |
| Outcomes | Production-grade CUDA on Blackwell; a real spatial-intelligence app in production |

---

## 2. Course description

Modern AI/ML engineering increasingly demands depth that pure-Python abstractions hide: hand-tuned CUDA kernels for new attention variants, quantization to FP8 on Blackwell tensor cores, classical multi-view geometry beneath learned 3D models, and the operational discipline to serve those models from both on-prem (DGX Spark) and cloud (SageMaker / Bedrock) targets.

This course takes you from **zero CUDA** to **shipping a fine-tuned NVIDIA Cosmos world model behind a NextJS application driven by a LangChain DeepAgent that perceives your home through real cameras and sensors**.

We do not optimize for portability.
We do not provide CPU-only fallbacks.
The bias is rigor and applied depth — at the cost of reproducibility for anyone without a Spark or comparable hardware.

---

## 3. Learning outcomes

By the end of the course you can:

1. **Write modern C++20** — concepts, ranges, RAII over CUDA handles, `std::span` / `std::mdspan` views, designated-initializer config structs — without copy-pasting from Stack Overflow.
2. **Author CUDA kernels from spec to profile-validated implementation**: coalesced access, shared-memory tiling, async copies, cooperative groups, tensor cores (WMMA / `mma.sync` / CUTLASS), and Blackwell features (TMA, thread-block clusters, FP8).
3. **Read a Nsight Compute report** and reason from `Speed of Light` through `Roofline` to a single, named bottleneck and a single, defensible refactor.
4. **Bridge classical and learned spatial intelligence**: sketch the 8-point algorithm and implement it on the GPU; train a 3D Gaussian splatting scene from your own captured video; fine-tune NVIDIA Cosmos and reason about the result.
5. **Ship to two production targets** with the same model: Triton on Spark (low-latency, sensor-coupled) and SageMaker BYOC or Bedrock custom-import on AWS (managed, scalable).
6. **Build a NextJS application backed by a LangChain DeepAgent** that ingests RTSP / RealSense streams, persists scenes to Postgres + pgvector, and answers grounded questions about a real space.

---

## 4. Required textbooks

You need physical or digital copies of all six.
Any book without a chapter table on the syllabus is consulted ad hoc; the books below are **load-bearing**.

| # | Short name | Full title | Author(s) | Role in course |
|---|---|---|---|---|
| 1 | **PMPP** | *Programming Massively Parallel Processors*, 4e | Hwu, Kirk, El Hajj | CUDA spine — Months 1-2 weekly readings |
| 2 | **Iglberger** | *C++ Software Design* | Iglberger | C++20 design idioms — Month 1 + ongoing |
| 3 | **PPP3** | *Programming: Principles and Practice Using C++*, 3e | Stroustrup | C++ language fundamentals — early Month 1 |
| 4 | **Torralba** | *Foundations of Computer Vision* | Torralba, Isola, Freeman | CV theory — Month 2 Week 8, Month 3 |
| 5 | **Hartley** | *Multiple View Geometry in Computer Vision*, 2e | Hartley & Zisserman | Geometry spine — Month 3 |
| 6 | **Szeliski** | *Computer Vision: Algorithms and Applications*, 2e | Szeliski | CV reference — Months 2-3 (consult more than read) |

**How to use them is its own document.**
See [`READING-GUIDE.md`](./READING-GUIDE.md).

---

## 5. Schedule (16 weeks)

This is a one-screen view.
The detailed plan with deliverables and perf targets lives in `.cursor/skills/curriculum-plan/month-N-*.md`.

### Month 1 — C++20 + CUDA Foundations

| Wk | Theme | Primary reading | Deliverable |
|---|---|---|---|
| 1 | Modern C++20 essentials + CUDA Hello | PMPP 1-3, Iglberger 1-3, PPP3 12-14 | `axpy<float>` ≥ 85% peak BW |
| 2 | Memory hierarchy + tiled GEMM | PMPP 4-5 | Four GEMM versions; tiled+async ≥ 50% cuBLAS |
| 3 | Reductions, scans, atomics | PMPP 6, 9-11 + Harris reduction | Reduction ≥ 80% `cub::DeviceReduce` |
| 4 | **Checkpoint** — production-grade SGEMM | PMPP 5-6 reread + CUTLASS intro | SGEMM ≥ **70% cuBLAS** at M=N=K=4096 |

### Month 2 — Advanced CUDA + CV Foundations

| Wk | Theme | Primary reading | Deliverable |
|---|---|---|---|
| 5 | Tensor cores (WMMA + CUTLASS) | PMPP 16, Blackwell whitepaper, CUTLASS docs | BF16 WMMA GEMM ≥ 60% cuBLAS |
| 6 | Fused attention from scratch | FlashAttention 1/2 (3 if Blackwell-specific) | BF16 attention ≥ 60% PyTorch SDPA |
| 7 | Multi-stream / graphs / (optional) NCCL | CUDA Guide §3.2.6, §3.2.8; PMPP 20 | Captured graph ≥ 1.4× serial |
| 8 | **Checkpoint** — GPU CV pipeline | Torralba 1-6, Hartley 1-3, 6 | 4K30 NVDEC → undistort → rectify pipeline |

### Month 3 — Multi-view Geometry + Cosmos

| Wk | Theme | Primary reading | Deliverable |
|---|---|---|---|
| 9 | Two-view geometry + 8-point on GPU | Hartley 9-11 | F-matrix + RANSAC ≤ 5 ms @ 2k corrs |
| 10 | Stereo + sparse SfM | Hartley 12-13, Szeliski 11-12, COLMAP paper | CUDA SGBM ≤ 25 ms @ 720p |
| 11 | NVIDIA Cosmos — fine-tune on custom data | Cosmos technical report + repo READMEs | Fine-tune completes < 24 hrs on Spark |
| 12 | **Checkpoint** — Gaussian splatting your scene | Kerbl et al SIGGRAPH 2023, Mip-Splatting | Trained scene ≥ 30 FPS @ 1080p, PSNR ≥ 25 dB |

### Month 4 — Production + DeepAgents

| Wk | Theme | Primary reading | Deliverable |
|---|---|---|---|
| 13 | TensorRT / TRT-LLM / FP8 on Blackwell | TensorRT Developer Guide; *FP8 for DL* | FP8 engine ≥ 2× PyTorch baseline |
| 14 | Dual deploy — Triton on Spark + SageMaker BYOC | Triton + SageMaker BYOC docs | Both endpoints p50 < 500 ms |
| 15 | NextJS + LangChain DeepAgents app | DeepAgents docs, Vercel AI SDK, Next.js 15 | Camera paired, agent answers grounded queries |
| 16 | **Capstone** — end-to-end home spatial-intel agent | Re-skim your own LAB.mds; pick 2-3 papers to disagree with | Public release; capstone writeup; ≥ 5 question types |

---

## 6. Daily loop in Cursor

The course is **taught** by Claude in Cursor.
Specifically: a roster of subagents plus a set of skills the agents know to consult.

### The loop, every day

```
read the assigned chapter(s)         ← brain on
   ↓
/start-week N (first day of week)    ← curriculum-mentor opens the week
   ↓
implement in labs/week-NN-*/         ← cuda-tutor + cpp20-tutor on standby
   ↓
build, run, test                     ← compute-sanitizer first
   ↓
/review-cuda                         ← cuda-code-reviewer reads the diff
   ↓
/profile-kernel                      ← cuda-perf-profiler runs the Nsight loop
   ↓
iterate (one variable per change)    ← keep diffs small
   ↓
/lab-report                          ← writes report/LAB.md from your code + reports
   ↓
/checkpoint (last day of week)       ← curriculum-mentor grades against rubric
```

### The roster (your office hours)

Treat each subagent as a senior colleague you can interrupt at zero cost.
Delegate aggressively — that's what they're for.

| Subagent | Bring to it… |
|---|---|
| `curriculum-mentor` | "Where am I?" / "Am I ready to advance?" / "Replan because I'm behind." |
| `cuda-tutor` | Kernel design, CUDA programming model, Blackwell features, CUTLASS choices |
| `cpp20-tutor` | Concepts, RAII, ranges, `std::span`/`mdspan`, design questions (Iglberger-style) |
| `spatial-intel-researcher` | Hartley/Szeliski/Torralba theory, Cosmos paper questions, "what's the modern paper for X?" |
| `dgx-spark-engineer` | "Why doesn't my container see the GPU?" / NGC, ARM64, NCCL between two Sparks |
| `cuda-perf-profiler` | "It works — is it fast?" / Nsight reports / roofline questions |
| `model-deployer` | TRT engine builds, Triton config, SageMaker BYOC, Bedrock custom-import |
| `langchain-deepagents-architect` | Month 4: agent shape, tools, NextJS streaming, sensor ingestion |
| `cuda-code-reviewer` | Diff review for correctness + perf + idiom + lab rigor |

### The slash commands

| Command | When |
|---|---|
| `/start-week N` | First action of a new week |
| `/review-cuda` | After substantive C++/CUDA edits |
| `/profile-kernel` | After tests pass; before claiming a perf number |
| `/lab-report` | When you're ready to write up a version |
| `/checkpoint` | End of week — mentor grades you |
| `/research-paper <topic>` | Side reading you want a citation-grade synthesis of |
| `/deploy-target {spark\|sagemaker\|bedrock}` | Months 4 (and earlier if you've earned it) |

### How to get the most out of the subagents

1. **Ask narrow, hard questions.** "Is `cudaMallocAsync` faster than `cudaMallocManaged` on Spark for my access pattern?" > "Tell me about memory."
2. **Give them files.** Mention the path; they'll read it. They are *not* general LLMs guessing — they ground in your code.
3. **Demand citations.** If the `cuda-tutor` doesn't cite a PMPP §, ask for it. Your habit is the rubric.
4. **Use `/review-cuda` early and often.** It's free and the reviewer has standards.
5. **Don't argue with `/checkpoint` results.** If the mentor returns `REWORK`, fix the highest-leverage thing and re-run. The point is the bar, not a passing grade.

---

## 7. Assessment & rubric

Every week is graded on five axes, 0-4 each:

| Axis | What we measure |
|---|---|
| **Correctness** | Tests pass at all sizes incl. edge cases; max-error within stated tolerance |
| **Performance** | Achieves the lab's stated perf target on Spark sm_121 |
| **Idiom** | C++20 modern (concepts/ranges/RAII), CUDA modern (cooperative groups, async copies, tensor cores) |
| **Profile evidence** | Nsight Systems trace + Nsight Compute report committed to `report/` |
| **Writeup** | `report/LAB.md` has hypothesis → method → results → next steps; cites primary sources |

| Week type | Pass threshold |
|---|---|
| Regular weeks (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15) | **14 / 20** |
| Checkpoint weeks (4, 8, 12) | **17 / 20** |
| Capstone (16) | **18 / 20** |

Full rubric details:
[`.cursor/skills/weekly-checkpoint/SKILL.md`](../.cursor/skills/weekly-checkpoint/SKILL.md).

---

## 8. Required artifacts per lab

Per `labs/week-NN-*/`:

- `LAB.md` — spec + readings + perf target (skeleton from `/start-week`)
- `src/` — C++ / CUDA implementation, modern idioms enforced
- `tests/` — GoogleTest with CPU reference, max-error assertion, edge sizes (1, prime, 2^28)
- `bench/` — microbenchmark reporting GB/s or GFLOPS vs roofline
- `report/`
  - `nsys_<kernel>.qdrep` — at least one
  - `ncu_<kernel>_v<N>.ncu-rep` — at least one
  - `LAB.md` — populated by `/lab-report`

No artifact, no claim.
A "fast kernel" without a Nsight Compute report is not fast — it's an assertion.

---

## 9. Academic integrity (the AI version)

You will use AI heavily — that is the point of the course.
The discipline is:

1. **Cite the AI's sources, not the AI.** If the `cuda-tutor` cites PMPP §5.4, your writeup cites PMPP §5.4. If it doesn't cite, you go read PMPP §5.4 yourself before quoting.
2. **Type the kernel.** Never paste an entire kernel from the agent without reading every line. The point is to understand it; the agent is a teacher, not a coworker who ships code.
3. **Profile with your own hands.** The agent can interpret reports. You generate them.
4. **Disagree out loud.** When you suspect an agent is wrong (it sometimes is, especially on the bleeding edge — Blackwell features and Cosmos releases move fast), say so in your `LAB.md` and check primary sources.

---

## 10. Stretch tracks (after Week 16)

If you finish ahead — and you might, if you skip some Weeks 1-3 material due to existing knowledge — pick a track:

- **Compiler track** — read the PTX ISA, write a Triton (OpenAI Triton) kernel for the same workload as a CUDA kernel you've written, and diff the SASS.
- **Distributed track** — pair two Sparks via ConnectX-7; run FSDP + expert parallelism on a 70B-class model.
- **Robotics track** — wire NVIDIA Isaac Lab into the Month-4 DeepAgent so the agent can act in simulation as well as perceive at home.

---

## 11. Office hours

The agents are available 24/7.
The fastest way to ask a question is to @-mention the relevant subagent in Cursor's chat with the file you're looking at attached.
If you don't know which subagent, ask the `curriculum-mentor` and it'll route you.

---

## 12. Capstone

End of Week 16:

1. **Working system.** ≥ 1 camera paired (RTSP or RealSense), trained Gaussian-splatting scene from Week 12 viewable in-app, fine-tuned Cosmos served from both Spark (Triton) and AWS, DeepAgent answers ≥ 5 grounded question types.
2. **Capstone writeup** (`labs/week-16-capstone/report/CAPSTONE.md`): abstract, system diagram, method, results (latency / throughput / cost), discussion, references.
3. **Public release.** Tag `v1.0.0`, polish the README, post.

This is the artifact you point recruiters at.

---

## 13. One-paragraph reading order

Weeks 1-4: PMPP 1-12, Iglberger 1-7, Stroustrup PPP3 4-7 + 12-22.
Weeks 5-8: PMPP 13-17, Iglberger 8-end, Torralba 1-6, Hartley 1-6.
Weeks 9-12: Hartley 9-15, Szeliski 11-13, Cosmos technical report, 3D Gaussian Splatting paper.
Weeks 13-16: TensorRT / TRT-LLM docs, Triton docs, SageMaker BYOC docs, Bedrock custom-model-import docs, LangChain DeepAgents docs, Next.js 15 + Vercel AI SDK docs.

Read first. Code second. Profile third. Write fourth.
