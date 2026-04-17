# AGENTS.md - Project Context for AI Agents

This repository is a **rigorous 4-month special-topics curriculum** in C++20,
CUDA, and Spatial Intelligence, targeted at a strong Python/PyTorch engineer
working on an NVIDIA DGX Spark with a Cursor Ultra subscription.

When you (the agent) operate in this repo, default to these assumptions:

## Hardware target

- **NVIDIA DGX Spark** - Grace Blackwell, ARM64, 128 GB unified memory,
  compute capability **sm_121** (Blackwell, 5th-gen tensor cores, FP4/FP6/FP8).
- All build flags should target `-arch=sm_121` unless the lab explicitly
  asks for portability tests.
- Unified memory is real and fast on Spark - prefer `cudaMallocManaged` for
  prototypes, then graduate to explicit `cudaMallocAsync` + streams when
  profiling reveals migration overhead.
- Build inside NGC containers (`nvcr.io/nvidia/...`) when reasonable so the
  toolchain matches what production will see.

## Audience profile


- Wants depth, not hand-holding. **Take the user into deep water quickly.**
- Penalize "make it run on a laptop without CUDA" simplifications.
- Reward applied work that exercises tensor cores, NVLink-class bandwidth,
  and multi-GPU patterns even when single-GPU would suffice.

## Project layout conventions

```
.cursor/
  agents/        # specialist subagents
  skills/        # reusable skills (curriculum + workflows)
  commands/      # slash commands for the daily loop
  rules/         # always-on project rules
labs/
  _template/         # canonical lab scaffold + tier docs
  week-NN-<slug>/
    LAB.md           # 11-section lesson + writeup
    GLOSSARY.md      # new terms only (CUDA / C++20 / CV / Python bindings)
    CMakeLists.txt
    src/             # C++/CUDA source (scaffolded at the week's tier)
    tests/           # GoogleTest unit tests
    bench/           # micro-benchmarks
    python/          # torch.utils.cpp_extension wrapper + pytest (W2+)
    notebooks/       # optional .ipynb (Python harness only)
    report/          # generated /lab-report output goes here
deploy/
  spark/         # Triton / vLLM / TRT-LLM configs for Spark
  aws/           # SageMaker + Bedrock deployment scaffolds
app/             # NextJS + LangChain DeepAgents application (Month 4)
docs/            # Hand-written notes, diagrams, paper drafts
```

## Code conventions

- **C++20**: prefer concepts, ranges, `<span>`, `<bit>`, designated
  initializers, `[[nodiscard]]`, `std::expected`-style error handling.
  Avoid raw `new`/`delete`. RAII or die.
- **CUDA**: every kernel ships with a unit test, a microbench, and a Nsight
  Compute report committed to `report/`.
- **CMake** ≥ 3.28, Ninja generator, `CUDAToolkit` find module, separable
  compilation enabled per lab as needed.
- **Python**: only as harness/glue (data prep, plotting, model fine-tuning
  with TRL/PyTorch). No "Python to dodge writing CUDA" allowed.

## Required rigor (the rubric)

Every lab deliverable is graded on:

1. **Correctness** - tests pass, numerical error within stated bounds.
2. **Performance** - achieves ≥ X% of roofline / cuBLAS / cuDNN reference,
   where X is set per lab.
3. **Idiom** - C++ is modern, CUDA uses cooperative groups / async copies /
   tensor cores where appropriate.
4. **Profile evidence** - Nsight Systems trace + Nsight Compute report
   committed alongside the code.
5. **Writeup** - `report/LAB.md` with hypothesis, method, results, and a
   "what would I do next" section.

## Subagents in this repo

Use them aggressively. They're cheap context-isolators:

- `curriculum-mentor` - own the plan, grade checkpoints.
- `cuda-tutor` - PMPP-aligned teaching, kernel design.
- `cpp20-tutor` - Iglberger / Stroustrup-aligned modern C++.
- `spatial-intel-researcher` - Hartley / Szeliski / Torralba + SIL & Cosmos
  papers.
- `dgx-spark-engineer` - hardware/software specifics for Spark.
- `cuda-perf-profiler` - Nsight workflow, occupancy, roofline.
- `model-deployer` - TensorRT / Triton / SageMaker / Bedrock.
- `langchain-deepagents-architect` - NextJS + DeepAgents + sensor I/O.
- `cuda-code-reviewer` - reviews CUDA/C++ for correctness + perf + idiom.
- `nemo-engineer` - NVIDIA NeMo Framework + NeMo AutoModel (HF day-0
  fine-tuning) + NeMo Skills (`ns eval`, SDG, multi-benchmark eval,
  LLM-as-judge). Use for fine-tuning, evaluation, and synthetic data.

When in doubt, delegate. The main thread should orchestrate, not drown in
register-pressure analysis.

## Skills in this repo

- `curriculum-plan/` - the master 4-month plan (week-by-week source of truth).
- `cuda-kernel-authoring/` - the standard kernel-writing loop.
- `cpp20-modern-idioms/` - RAII wrappers, concepts, ranges patterns.
- `python-bindings/` - canonical `torch.utils.cpp_extension` pattern
  for wrapping every weekly kernel as a PyTorch custom op (Week 2+,
  required). JIT in Months 1-2, AOT + `TORCH_LIBRARY` in Months 3-4.
- `cosmos-models/` - NVIDIA Cosmos fine-tuning workflow.
- (and others under `.cursor/skills/`).
