# Month 4 — Production + LangChain DeepAgents Capstone

Goal: by end of Month 4 you have a NextJS application driven by a
LangChain DeepAgent that perceives a real scene through cameras/sensors
in your home, reasons about it with a fine-tuned NVIDIA Cosmos model
served through Triton on your Spark **and** through a SageMaker (or
Bedrock-imported) endpoint in AWS, and answers grounded questions.

> **Standing learning objective (every week of Month 4).** Any custom
> CUDA kernel in the production stack must ship as a PyTorch custom
> op via `torch.utils.cpp_extension` (AOT — `setup.py` +
> `BuildExtension`, never JIT in production), registered as a
> `TORCH_LIBRARY` op so it survives `torch.compile` and TensorRT
> export. `pytest` numerics gate: max-abs-error against CPU reference
> within tolerance, and wrapper overhead < 5% of kernel time at
> production batch size. Pattern documented in
> `.cursor/skills/python-bindings/SKILL.md`.

---

## Week 13 — Inference optimization (TensorRT, TRT-LLM, FP8 on Blackwell)

**Theme.** Take a model that "trained successfully" and turn it into
something a product can call. You will quantize, build a TensorRT
engine, and benchmark vs the PyTorch baseline.

**Readings.**
- TensorRT Developer Guide (latest) — §1-4, §10 (quantization).
- TensorRT-LLM docs — *quick start*, *FP8 on Blackwell*, *KV cache
  reuse*.
- Blackwell whitepaper — quantization formats (E4M3, E5M2, MXFP4/MXFP6).
- *FP8 Formats for Deep Learning* (Micikevicius et al, NVIDIA).

**Lab — `labs/week-13-trt-optimize/`.**
1. Export your fine-tuned Cosmos variant from Week 11 to ONNX (or use
   the TRT-LLM checkpoint conversion path if it's a transformer).
2. Build TRT engines at FP16, BF16, and FP8. For each, measure latency
   (p50/p95) and accuracy delta on a held-out set.
3. Use CUDA Graphs (`--useCudaGraph`) and KV-cache reuse where
   applicable.
4. Compare to the PyTorch baseline.

**Performance target.** FP8 engine ≥ 2× PyTorch baseline throughput
with accuracy delta within 1% of FP16.

---

## Week 14 — Dual deploy: Triton on Spark + SageMaker BYOC endpoint

**Theme.** Same model, two targets. Production sees both.

**Readings.**
- Triton Inference Server docs — *model repository*, *dynamic batching*,
  *backends* (TRT, Python, TRT-LLM).
- SageMaker docs — *bring your own container*, *real-time inference*,
  *async endpoints* (which may be a better fit if model latency is
  high).
- Bedrock — *custom model import* (verify current support for your
  model architecture; if unsupported, document the limitation).

**Lab — `labs/week-14-dual-deploy/`.**
1. Stand up Triton on Spark with your TRT engine in `deploy/spark/model_repo/`.
   Health-check, `perf_analyzer`-bench at concurrency 1, 4, 16.
2. Build a SageMaker BYOC container that wraps the same model. Note
   that you must rebuild the TRT engine targeting `sm_90` (Hopper,
   `ml.p5*`) or `sm_89` (Ada, `ml.g6*`) inside the SageMaker container —
   Spark's `sm_121` engine does not transfer.
3. Deploy a real-time SageMaker endpoint via the Python SDK; smoke-test
   `boto3.client("sagemaker-runtime").invoke_endpoint(...)`.
4. Write a small Python harness that benchmarks both targets at the
   same workload and reports cost/$1k inferences.

**Performance target.** Both endpoints respond `<` 500 ms p50 for the
canonical request shape; cost-per-1k clearly compared.

---

## Week 15 — NextJS + LangChain DeepAgents app, sensor ingestion

**Theme.** Build the user-facing app. The DeepAgent has tools that:
1. perceive (call your perception endpoint),
2. reason (call your reasoning endpoint),
3. retrieve (vector-search over prior scenes),
4. switch compute target (Spark vs AWS).

**Readings.**
- Deep Agents docs (`docs.langchain.com/oss/python/deepagents`) — *Quickstart*,
  *Subagents*, *Backends* (use Postgres/LangGraph store), *Skills*.
- Vercel AI SDK docs — `useChat`, `streamText`, server-sent events.
- Next.js 15 App Router — *Route Handlers*, *Streaming*, *Server Actions*.
- `mediamtx` docs for RTSP → WebRTC.
- `pyrealsense2` docs if you're using a RealSense.

**Lab — `labs/week-15-nextjs-deepagent/` and `app/`.**
1. Stand up `app/` (Next.js 15, TypeScript, Tailwind v4, shadcn/ui).
2. Stand up `agent/` (Python FastAPI service running `deepagents`).
3. Implement the tools listed above. Persist scenes to Postgres +
   pgvector.
4. Wire a chat UI in NextJS that streams the agent's tokens AND its
   tool calls (so the user sees "calling perceive_scene…").
5. Pair at least one real camera (RTSP IP cam or RealSense) and prove
   the perception tool returns frames.

**Performance target.** Round-trip "perceive + reason + answer" latency
≤ 3 s on Spark for a 1-second window of 720p frames.

---

## Week 16 — Capstone: end-to-end home spatial-intelligence agent

**Theme.** Ship it. Write it up paper-style. Open-source it.

**Readings (light, mostly synthesis).**
- Re-skim your own `report/LAB.md`s from Weeks 1-15. The capstone
  writeup is the throughline of all of them.
- Pick 2-3 papers you cited that you now disagree with, and write
  *why* you disagree (rare, valuable, top-percentile move).

**Lab — `labs/week-16-capstone/` plus polished `app/`.**
1. End-to-end demo:
   - At least one camera paired (RTSP or RealSense).
   - Trained Gaussian-splatting scene from Week 12 viewable in-app.
   - Fine-tuned Cosmos served from both Spark (Triton) and AWS
     (SageMaker).
   - DeepAgent answers at least 5 grounded questions about your scene
     ("did anyone enter the kitchen in the last hour?", "show me
     where the toolbox was at 3pm yesterday", "what changed since
     8am?", etc.) with timestamped frame references.
2. Capstone writeup `report/CAPSTONE.md`:
   - Abstract (200 words).
   - System diagram.
   - Method (per major component, what you built and why).
   - Results (latency, throughput, cost, qualitative samples).
   - Discussion (what surprised you, what's brittle, what's next).
   - Acknowledgments (the books and papers; cite properly).
3. README polish for public release; tag `v1.0.0`.

**Performance target (capstone — strict).**
- DeepAgent end-to-end latency ≤ 3 s on Spark, ≤ 5 s on AWS.
- ≥ 5 question types working.
- All 16 weeks' code compiles, all tests pass on Spark, all
  reports committed.

**Checkpoint rubric.** Capstone — needs **18/20**. This is the artifact
you point recruiters at.
