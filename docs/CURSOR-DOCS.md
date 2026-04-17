# Cursor Docs to Import

Cursor lets you index third-party documentation and reference it in chat with `@<DocName>`.
The agents in this repo expect a specific catalogue of docs to be available, so they can cite primary sources ("see `@PTX` §9.7.13 on `cp.async`...") instead of paraphrasing from training data.

This page lists every doc the curriculum assumes is imported, why it matters, and which week / agent will reach for it.
**Import them once before Week 1.**
They take a few minutes each to crawl and the investment pays off all 16 weeks.

---

## How to import a doc into Cursor

1. Open Cursor settings → **Indexing & Docs** (or click `+ Add Doc` in the Docs panel).
2. Paste the doc's root URL (the ones in the tables below).
3. Give it a short, memorable **name** (the names below match what the agents and prompts assume — e.g. `PTX`, not `nvidia-ptx-isa`).
4. Wait for the crawl to finish (status flips to "Indexed").
5. Reference in chat with `@<name>` — e.g. `@PTX`, `@C++20`, `@NeMo Skills`.

Re-index a doc whenever you see it's stale (right-click → re-index).
NVIDIA's docs change with every CUDA release; refresh after each.

---

## 1. Core — import these first (Months 1-2)

The non-negotiables.
Without these, the CUDA / C++ tutors will hand-wave instead of citing.

| Cursor name | Source URL | Why this curriculum needs it | Used by |
|---|---|---|---|
| **PTX** | <https://docs.nvidia.com/cuda/parallel-thread-execution/> | The PTX ISA reference — what `nvcc` generates, what Nsight Compute shows you. Required for any "what does this SASS line mean?" question. | `cuda-tutor`, `cuda-perf-profiler`, `cuda-code-reviewer` |
| **C++20** | <https://en.cppreference.com/w/cpp/20.html> | The single source of truth for language + library features. Concepts, ranges, `<span>`, `<bit>`, `<format>`, `std::expected`-style patterns. | `cpp20-tutor`, `cuda-code-reviewer` |
| **NVIDIA DGX Spark** | <https://docs.nvidia.com/dgx/> (Spark section) | Hardware specifics: sm_121, unified memory, ConnectX-7, DGX OS quirks. | `dgx-spark-engineer`, every kernel decision |
| **CUDA CCL** | <https://nvidia.github.io/cccl/> | The CUDA C++ Core Libraries (Thrust, CUB, libcudacxx). Reach here before writing your own scan / reduce / sort. | `cuda-tutor`, `cuda-code-reviewer` |
| **cuBLAS** | <https://docs.nvidia.com/cuda/cublas/> | Reference GEMM — your performance bar in Weeks 2 + 6. | `cuda-tutor`, `cuda-perf-profiler` |
| **cuDNN** | <https://docs.nvidia.com/deeplearning/cudnn/api/> | Reference attention / convolution kernels for your fused-attention week. | `cuda-tutor`, `cuda-perf-profiler` |
| **NVIDIA cuTile** | <https://developer.nvidia.com/blog/cutile/> *(or current docs page)* | Tile-programming abstractions on Blackwell — the abstraction layer above raw CUDA for many tensor-core workloads. | `cuda-tutor`, Month 2 tensor-core labs |

---

## 2. Inference + serving (Months 3-4)

Once you have models, you need to serve them.
These cover both on-Spark and AWS paths.

| Cursor name | Source URL | Why | Used by |
|---|---|---|---|
| **TensorRT** | <https://docs.nvidia.com/deeplearning/tensorrt/> | The optimization layer for any model going to production on NVIDIA hardware. | `model-deployer` |
| **TensorRT-LLM** | <https://nvidia.github.io/TensorRT-LLM/> | LLM-specific TRT path. The default for serving fine-tuned LLMs on Spark. | `model-deployer`, `nemo-engineer` |
| **vLLM** | <https://docs.vllm.ai/> | Fast iteration server — the right choice during fine-tune → eval loops before you build a TRT engine. | `nemo-engineer`, `model-deployer` |
| **NVIDIA Dynamo** | <https://docs.nvidia.com/dynamo/> | NVIDIA's distributed inference framework — the `nvidia-smi` of multi-node serving. Use for Spark-cluster (2× ConnectX-7) experiments. | `model-deployer` |
| **AWS Bedrock** | <https://docs.aws.amazon.com/bedrock/> | Cloud target for `/deploy-target bedrock`. Custom-model import + Foundation Model API. | `model-deployer` |

> **Note.** SageMaker is the other AWS target but doesn't need a dedicated import — `model-deployer` cites the AWS SageMaker docs dynamically.
> Add it explicitly if you want consistent `@SageMaker` referencing.

---

## 3. NVIDIA NeMo stack (Months 3-4 fine-tuning + eval)

The whole NeMo family.
The `nemo-engineer` agent is built around these.

| Cursor name | Source URL | Why | Used by |
|---|---|---|---|
| **NeMo** | <https://docs.nvidia.com/nemo-framework/> | The NeMo Framework umbrella — LLMs, ASR, TTS, multimodal. Parallelism, PTQ/QAT, sequence packing. | `nemo-engineer` |
| **NeMo AutoModel** | <https://docs.nvidia.com/nemo/automodel/latest/> | Day-0 fine-tuning for any HF causal LM / VLM. The default fine-tune path. | `nemo-engineer` |
| **NeMo Skills** | <https://nvidia-nemo.github.io/Skills/> | `ns eval`, SDG, multi-benchmark eval, LLM-as-judge. The Months 3-4 eval gate. | `nemo-engineer`, `/checkpoint`, `/deploy-target` |
| **NeMo RL** | <https://docs.nvidia.com/nemo/rl/> *(or NVIDIA-NeMo/RL on GitHub Pages)* | DPO / GRPO / RM pipelines downstream of AutoModel checkpoints. | `nemo-engineer` |
| **NeMo Curator** | <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/> | Data preparation at scale — dedupe, quality filter, language ID. Pair with NeMo Skills SDG. | `nemo-engineer` |
| **NeMo Evaluator** | <https://docs.nvidia.com/nemo/evaluator/> | Standalone evaluation service. Good when NeMo Skills is overkill for a one-off eval. | `nemo-engineer` |
| **NeMo Agent Toolkit** | <https://docs.nvidia.com/nemo/agent-toolkit/> | NVIDIA's agent-building primitives. Useful as a comparison point against LangChain DeepAgents in Month 4. | `langchain-deepagents-architect` (compare/contrast) |

---

## 4. Spatial Intelligence + Cosmos (Month 3)

The applied target of the curriculum.

| Cursor name | Source URL | Why | Used by |
|---|---|---|---|
| **NVIDIA Cosmos** | <https://docs.nvidia.com/cosmos/> | Cosmos Predict / Reason / Transfer world foundation models. Month 3 fine-tune target. | `spatial-intel-researcher`, `nemo-engineer` |
| **NVIDIA Spatial Intelligence Lab** | <https://research.nvidia.com/labs/sil/> | The lab's research output — papers, demos, model cards. Direct primary-source citations for `/research-paper`. | `spatial-intel-researcher` |

---

## 5. Month-4 application stack (NextJS + agent + ops)

When the curriculum lands in production.

| Cursor name | Source URL | Why | Used by |
|---|---|---|---|
| **LangChain DeepAgents** | <https://docs.langchain.com/labs/deep-agents/> | The agent framework powering the Month-4 NextJS app. | `langchain-deepagents-architect` |
| **LangChain AWS** | <https://python.langchain.com/docs/integrations/platforms/aws/> | Bedrock / SageMaker / S3 integrations. The bridge between your DeepAgent and the AWS deploy target. | `langchain-deepagents-architect`, `model-deployer` |
| **LangSmith** | <https://docs.smith.langchain.com/> | Tracing + eval for your agent. The "Nsight Compute" of LLM agents. | `langchain-deepagents-architect` |
| **Traefik** | <https://doc.traefik.io/traefik/> | Reverse proxy / ingress for the NextJS + Triton + agent stack on Spark. The clean way to expose `app.local` over TLS. | `dgx-spark-engineer`, `langchain-deepagents-architect` |

---

## 6. Visualization + teaching aids

*This curriculum no longer ships dedicated visualization agents.
Static `docs/*.excalidraw` files (e.g. [`SYLLABUS.excalidraw`](./SYLLABUS.excalidraw), [`READING-GUIDE.excalidraw`](./READING-GUIDE.excalidraw)) are maintained by hand using Cursor's Excalidraw extension.
No Cursor doc imports are required for visualization.*

---

## 7. The `@`-references this curriculum assumes

When an agent (or this codebase's prompts) writes `@PTX`, it expects the doc to be importable.
Below is the canonical name → tool mapping so you can verify your imports match what the prompts use.

| Agent / command | Will reach for |
|---|---|
| `cuda-tutor`, `/review-cuda` | `@PTX`, `@CUDA CCL`, `@cuBLAS`, `@cuDNN`, `@NVIDIA cuTile` |
| `cpp20-tutor`, `/review-cuda` | `@C++20` |
| `dgx-spark-engineer` | `@NVIDIA DGX Spark`, `@Traefik` (when ingress is in scope) |
| `cuda-perf-profiler`, `/profile-kernel` | `@PTX`, `@cuBLAS`, `@cuDNN` |
| `spatial-intel-researcher`, `/research-paper` | `@NVIDIA Cosmos`, `@NVIDIA Spatial Intelligence Lab` |
| `nemo-engineer` | `@NeMo`, `@NeMo AutoModel`, `@NeMo Skills`, `@NeMo RL`, `@NeMo Curator`, `@NeMo Evaluator`, `@NeMo Agent Toolkit` |
| `model-deployer`, `/deploy-target` | `@TensorRT`, `@TensorRT-LLM`, `@vLLM`, `@NVIDIA Dynamo`, `@AWS Bedrock`, `@LangChain AWS` |
| `langchain-deepagents-architect` | `@LangChain DeepAgents`, `@LangChain AWS`, `@LangSmith`, `@NeMo Agent Toolkit` |

---

## 8. Habits that compound

- **Cite, don't paraphrase.** When you ask an agent a "why" question, end with "cite `@PTX` §X.Y". The good agents already do this; the habit reinforces it.
- **Re-index after big releases.** CUDA Toolkit, cuDNN, TensorRT-LLM, NeMo, and vLLM ship breaking changes every few months. A stale doc will quietly mislead you. Re-index quarterly.
- **Don't import what you won't use.** Each indexed doc costs Cursor storage + retrieval time. The list above is *the* list — adding twenty more rarely-used docs degrades retrieval quality on the ones that matter.
- **One name per source.** If you renamed `NVIDIA NeMo Skills` to `nemo-skills` locally, every prompt in this repo that says `@NeMo Skills` will miss. Match the names in §1-§6 exactly.
- **When you add a doc, add a row here.** This file is the index of truth. Drift between what's imported and what the agents assume is a quiet productivity tax.

---

## 9. Quick checklist

Before `/start-week 1`, confirm these are all `Indexed` in your Docs panel:

- [ ] `PTX`
- [ ] `C++20`
- [ ] `NVIDIA DGX Spark`
- [ ] `CUDA CCL`
- [ ] `cuBLAS`
- [ ] `cuDNN`
- [ ] `NVIDIA cuTile`
- [ ] `TensorRT`
- [ ] `TensorRT-LLM`
- [ ] `vLLM`
- [ ] `NVIDIA Dynamo`
- [ ] `AWS Bedrock`
- [ ] `NeMo`
- [ ] `NeMo AutoModel`
- [ ] `NeMo Skills`
- [ ] `NeMo RL`
- [ ] `NeMo Curator`
- [ ] `NeMo Evaluator`
- [ ] `NeMo Agent Toolkit`
- [ ] `NVIDIA Cosmos`
- [ ] `NVIDIA Spatial Intelligence Lab`
- [ ] `LangChain DeepAgents`
- [ ] `LangChain AWS`
- [ ] `LangSmith`
- [ ] `Traefik`

That's 25 docs.
Twenty minutes of crawling, sixteen weeks of citation-grade answers.
