# CUDA Spatial Intelligence Lab

A 4-month, deep-water special-topics curriculum that takes a strong Python/PyTorch
engineer from zero to advanced in **C++20 + CUDA + Spatial Intelligence**, then
ships a real product: a NextJS app driven by a LangChain DeepAgent that talks
to home cameras/sensors and runs Spatial Intelligence models served from an
NVIDIA DGX Spark or AWS (SageMaker / Bedrock).

> **Audience.** You want top-percentile AI/ML Engineer skills. 
> You own a DGX Spark, a Cursor Ultra subscription, and an AWS account. 
> You don't want toy reproductions; you want production CUDA on
> Grace Blackwell with a real spatial agent on top.
>
> **Bias.** Rigor over reproducibility. Depth over breadth.

---

## What this repo gives you

This is **not** a textbook clone. It is a Cursor-native curriculum harness:

- **`.cursor/skills/curriculum-plan/`** - the master 4-month plan, broken into
  16 weeks with readings, labs, deliverables, and checkpoint rubrics.
- **`.cursor/agents/`** - 10 specialist subagents (curriculum mentor,
  CUDA tutor, C++20 tutor, DGX Spark engineer, Nsight profiler,
  Spatial Intelligence researcher / PhD-advisor paper grader, NeMo
  engineer, model deployer, DeepAgents architect, CUDA code reviewer).
- **`.cursor/skills/`** - reusable workflows (kernel authoring, profiling,
  Cosmos fine-tuning, AWS endpoint deploy, lab notebook authoring, etc.).
- **`.cursor/commands/`** - slash commands for the daily loop
  (`/start-week N`, `/lab-report`, `/review-cuda`, `/profile-kernel`,
  `/checkpoint`, `/research-paper`, `/deploy-target`).
- **`.cursor/rules/`** - always-on project context (hardware, conventions,
  required rigor).
- **`labs/week-XX-*/`** - one lab folder per week, with starter `CMakeLists.txt`,
  source skeletons, and a `LAB.md` defining objectives, references, and the
  rubric you'll be graded against.

## Student docs

- **[`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md)** — **start
  here**. Verify your driver, CUDA toolkit, host C++ compiler, CMake,
  and NGC container setup. Includes `nvidia-smi` and `nvcc` cheat
  sheets and a smoke test.
- **[`docs/SYLLABUS.md`](docs/SYLLABUS.md)** — the modern syllabus:
  outcomes, schedule, daily loop in Cursor, rubric, capstone.
- **[`docs/READING-GUIDE.md`](docs/READING-GUIDE.md)** — how to use the
  six textbooks (chapter maps, reading modes, when to consult which
  subagent).
- **[`docs/WORKFLOW.md`](docs/WORKFLOW.md)** — how to actually drive
  the slash commands and partner with the ten subagents to get
  through the 16 weeks. The four cycles, the escalation tree, the
  weekly rhythm.
- **[`docs/CURSOR-DOCS.md`](docs/CURSOR-DOCS.md)** — the canonical
  list of third-party docs to import into Cursor's `@Docs` index
  (PTX, C++20, NeMo, TensorRT-LLM, Cosmos, LangChain DeepAgents, …).
  The agents in this repo assume these names exist. Import once
  before Week 1.

## Required reading (own physical or digital copies)

| # | Title | Author | Use |
|---|---|---|---|
| 1 | *Programming Massively Parallel Processors* (4e) | Hwu, Kirk, El Hajj | CUDA spine |
| 2 | *C++ Software Design* | Iglberger | Modern C++20 design idioms |
| 3 | *Programming: Principles and Practice Using C++* (3e) | Stroustrup | C++ foundations |
| 4 | *Foundations of Computer Vision* | Torralba, Isola, Freeman | CV theory spine |
| 5 | *Multiple View Geometry in Computer Vision* (2e) | Hartley & Zisserman | Geometry spine |
| 6 | *Computer Vision: Algorithms and Applications* (2e) | Szeliski | CV reference |

## Hardware / software baseline

- **NVIDIA DGX Spark** (Grace Blackwell, 128 GB unified memory, ARM64, sm_121).
- **NVIDIA DGX OS** + NVIDIA Container Runtime for Docker.
- **CUDA Toolkit 13+**, **GCC 13+** (or Clang 17+), **CMake 3.28+**, **Ninja**.
- **Nsight Systems / Nsight Compute** (ARM64 builds via NGC).
- **NGC** account + Docker login to `nvcr.io`.
- **AWS** account with SageMaker + Bedrock access in a region near you.
- **Cursor Ultra** with this repo opened. The skills/agents/commands light up
  automatically.

## The 4-month arc

| Month | Theme | Key outputs |
|---|---|---|
| 1 | C++20 + CUDA foundations | Tiled GEMM, parallel reduction, custom RAII tensor |
| 2 | Advanced CUDA + CV foundations | Tensor-core GEMM, fused attention kernel, multi-stream pipeline, GPU rectification |
| 3 | Multi-view geometry + Cosmos | 8-point on GPU, stereo, fine-tuned NVIDIA Cosmos world model, Gaussian splatting scene |
| 4 | Production + DeepAgents | TensorRT-optimized Cosmos, dual deploy (Spark + SageMaker/Bedrock), NextJS + LangChain DeepAgent driving real sensors |

See [`.cursor/skills/curriculum-plan/SKILL.md`](.cursor/skills/curriculum-plan/SKILL.md)
for the full week-by-week plan.

## Daily loop in Cursor

1. `/start-week N` - the curriculum-mentor subagent prints week N's objectives,
   readings, and deliverables, then queues subagents you'll need.
2. Work in `labs/week-NN-*/`. Use `/review-cuda` after substantive kernel edits.
3. `/profile-kernel` to drive the Nsight workflow when a kernel "works".
4. `/lab-report` to generate the week's `LAB.md` writeup.
5. `/checkpoint` at end of week - mentor grades you against the rubric.

## Additional Resources

GPU Mode community: https://www.gpumode.com/home 
Modal AI GPU Glossary: https://modal.com/gpu-glossary 