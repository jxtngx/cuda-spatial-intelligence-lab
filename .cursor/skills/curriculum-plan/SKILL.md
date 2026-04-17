---
name: curriculum-plan
description: The master 4-month cuda-spatial-intelligence-lab curriculum. Use when the user asks about the overall plan, what week they are on, what to study next, or for the rubric used to grade weekly checkpoints. Read the relevant month file for week-level detail.
---

# 4-Month Curriculum: CUDA + C++20 + Spatial Intelligence

## Audience and intent

A strong Python/PyTorch engineer with deep ML-systems experience goes from
zero to advanced in C++20 and CUDA on an NVIDIA DGX Spark, lands in
Spatial Intelligence (NVIDIA Cosmos + Gaussian splatting + multi-view
geometry), and ships a NextJS app driven by a LangChain DeepAgent that
talks to home cameras/sensors and serves models from Spark or AWS.

**Bias:** rigor over reproducibility. Depth over breadth. Applied over
academic, but every applied lab is anchored to a primary text or paper.

## Time budget

- ~20 hrs/week of focused work (3 hrs/weekday + a long weekend session).
- 4 months = 16 weeks = 16 labs.
- Weeks 4, 8, 12, 16 are integration/checkpoint weeks where the deliverable
  is harder and the rubric is stricter.

## The arc

| Month | Theme | Big deliverable |
|---|---|---|
| 1 | C++20 + CUDA foundations | Tiled FP32 GEMM hitting ≥ 70% of cuBLAS on Spark |
| 2 | Advanced CUDA + CV foundations | Tensor-core (BF16) GEMM ≥ 60% of cuBLAS, fused attention kernel, GPU image rectification |
| 3 | Multi-view geometry + Cosmos | 8-point algorithm on GPU, fine-tuned Cosmos-Predict on a custom video dataset, Gaussian-splatting scene reconstruction |
| 4 | Production + DeepAgents | Cosmos served via TensorRT-LLM + Triton on Spark, dual-target deploy (Spark + SageMaker/Bedrock), NextJS DeepAgent answering grounded questions about a real home scene |

## Weekly index

### Month 1 — Foundations (`month-1-foundations.md`)
- **Week 1**: Modern C++20 essentials + CUDA Hello (PMPP Ch 1-3, Iglberger Ch 1-3, Stroustrup PPP3 selected)
- **Week 2**: Memory hierarchy + tiled GEMM (PMPP Ch 4-5)
- **Week 3**: Reductions, scans, atomics, warp primitives (PMPP Ch 6, 10, 11)
- **Week 4 (checkpoint)**: GEMM target — ≥ 70% of cuBLAS FP32

### Month 2 — Advanced CUDA + CV foundations (`month-2-cuda-advanced.md`)
- **Week 5**: Tensor cores via WMMA + CUTLASS (PMPP Ch 16, CUTLASS docs)
- **Week 6**: Fused attention kernel from scratch (FlashAttention paper)
- **Week 7**: Multi-stream / async / `cuda::pipeline` + (optional) two-Spark NCCL
- **Week 8 (checkpoint)**: CV foundations — Torralba Ch 1-6, Hartley §1-6; lab: GPU image rectification + undistortion at 4K30

### Month 3 — Multi-view geometry + Cosmos (`month-3-spatial-intelligence.md`)
- **Week 9**: Two-view geometry + 8-point on GPU (Hartley §11)
- **Week 10**: Stereo + simple SfM (Hartley §11-13, Szeliski §11-12)
- **Week 11**: NVIDIA Cosmos-Predict / Cosmos-Reason — fine-tune on a custom dataset
- **Week 12 (checkpoint)**: Train a Gaussian-splatting scene from your own captured data; render at ≥ 30 FPS at 1080p on Spark

### Month 4 — Production + DeepAgents (`month-4-production-deepagents.md`)
- **Week 13**: TensorRT / TensorRT-LLM optimization, FP8 quantization on Blackwell
- **Week 14**: Dual deploy — Triton on Spark + SageMaker (BYOC) endpoint with the same model
- **Week 15**: NextJS + LangChain DeepAgents app, sensor ingestion (RTSP / RealSense)
- **Week 16 (capstone)**: End-to-end home spatial-intelligence agent, written up paper-style

## The rubric (every lab graded on this)

5 axes, 0–4 each, **20 total, 14 to advance**. Weeks 4/8/12/16 require **17**.

| Axis | 4 (Excellent) | 2 (Pass) | 0 (Fail) |
|---|---|---|---|
| Correctness | Tests pass at all sizes incl. edge cases; max-error within 2× theoretical | Tests pass at the lab's required sizes | Tests fail or no tests exist |
| Performance | Exceeds the lab's perf target | Hits the perf target | Below 50% of target |
| Idiom | C++20 (concepts/ranges/RAII), CUDA (cooperative groups, async copies, tensor cores where relevant) | Modern but not exemplary | Raw `new`, no concepts, blocking memcpys |
| Profile evidence | Nsight Systems trace + Nsight Compute report committed; bottleneck named | One report committed | No profiling artifacts |
| Writeup | `report/LAB.md` with hypothesis → method → results → next steps; cites primary sources | Has results | Missing or shallow |

## Daily loop in Cursor

1. `/start-week N` → curriculum-mentor opens the week.
2. Read the assigned chapters first. Do not write code before reading.
3. Implement in `labs/week-NN-*/`. Use `cpp20-tutor` and `cuda-tutor` as
   you go.
4. After it compiles, `/review-cuda`.
5. After it passes tests, `/profile-kernel`.
6. After perf target hit, `/lab-report`.
7. End of week: `/checkpoint`.

## Stretch tracks

If you blow through the schedule:
- **Compiler track**: read the PTX ISA + write a Triton kernel and compare
  to your hand CUDA.
- **Distributed track**: pair two Sparks via ConnectX-7, run FSDP on a 70B
  model with expert parallelism.
- **Robotics track**: integrate Isaac Lab simulator and have the Month-4
  DeepAgent control a simulated robot in addition to home sensors.

## Reading order quick-reference

Weeks 1-4: PMPP Ch 1-12, Iglberger Ch 1-7, Stroustrup PPP3 Ch 4-7, 12-22.
Weeks 5-8: PMPP Ch 13-17, Iglberger Ch 8-end, Torralba Ch 1-6, Hartley §1-6.
Weeks 9-12: Hartley §11-15, Szeliski §11-13, Cosmos paper(s), 3DGS paper.
Weeks 13-16: TensorRT-LLM docs, Triton docs, SageMaker BYOC docs, Bedrock
custom-model-import docs, LangChain DeepAgents docs, Next.js 15 + Vercel
AI SDK docs.
