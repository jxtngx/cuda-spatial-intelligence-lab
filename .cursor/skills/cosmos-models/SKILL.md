---
name: cosmos-models
description: Working with NVIDIA Cosmos World Foundation Models (Cosmos-Predict, Cosmos-Reason, Cosmos-Transfer) on DGX Spark — pulling containers, running baseline inference, fine-tuning on custom data, and exporting for production. Use when the user is on Week 11+ or otherwise interacting with Cosmos.
---

# NVIDIA Cosmos on DGX Spark

NVIDIA Cosmos is a family of *world foundation models* for physical AI — they
predict, reason about, and transfer video of physical scenes. On a Spark
you have enough unified memory (128 GB) to inference and fine-tune
mid-size variants comfortably.

## Family at a glance

- **Cosmos-Predict** — autoregressive / diffusion world model. Given a
  prompt (image, video, or text), predicts plausible future frames.
- **Cosmos-Reason** — VLM-class reasoner over video; answers questions
  about a clip with grounded references.
- **Cosmos-Transfer** — controllable video generation conditioned on
  segmentation, depth, or text.
- **Cosmos-Tokenizer** — the spatial-temporal tokenizer used by the
  others; useful as a standalone embedding model.

(Names and exact lineup move with releases — confirm against the current
Cosmos GitHub README before quoting.)

## Authoritative sources

- **Cosmos GitHub** — `https://github.com/NVIDIA/Cosmos` (and the
  `Cosmos-Predict*`, `Cosmos-Reason*` repos).
- **Cosmos technical report** (most recent on arXiv).
- **NGC catalog** — `nvcr.io/nvidia/cosmos*` containers and model cards.
- **Hugging Face** — `nvidia/Cosmos-*` model repos.

## Setup on Spark

```bash
docker pull nvcr.io/nvidia/cosmos-predict:<tag>     # confirm tag from Cosmos repo
docker pull nvcr.io/nvidia/cosmos-reason:<tag>

mkdir -p ~/cosmos/{checkpoints,data,outputs}

# Hugging Face token if pulling weights directly:
export HF_TOKEN=hf_...

# Run a container with unified memory + your scratch dirs mounted:
docker run --gpus all --rm -it \
  --shm-size=32g --ipc=host \
  -v ~/cosmos:/workspace/cosmos \
  -e HF_TOKEN \
  nvcr.io/nvidia/cosmos-predict:<tag> bash
```

Inside the container, follow the repo's README to download weights into
`/workspace/cosmos/checkpoints`.

## Baseline inference smoke test

Pick the smallest variant first. Confirm:

1. Weights load (no OOM — should be fine on Spark even for mid-size).
2. A 5-second 720p sample completes in expected wall time on `sm_121`.
3. Output is sensible against a known prompt from the model card.

If you OOM, you almost certainly pulled an x86 image or skipped
`--shm-size`/`--ipc=host`.

## Fine-tuning workflow (Week 11)

1. **Curate ≥ 200 clips** of your target domain (~5-10 s each, ≥ 720p).
   Provide captions if the variant expects text conditioning.
2. **Preprocess** with the Cosmos tokenizer into the model's expected
   format. Cache the tokenized tensors to disk to save GPU time across
   epochs.
3. **Run TRL / the official Cosmos fine-tuning script**:
   - For diffusion variants: LoRA on the U-Net / DiT blocks first;
     consider full fine-tune only if you have a clear signal.
   - For reasoning variants: SFT on `(video, question, answer)` triples.
4. **Log to Trackio** (`hugging-face-trackio` skill if you want help) so
   loss curves are inspectable.
5. **Eval qualitatively** on held-out clips first; quantitative metrics
   (FVD, etc.) only if your dataset supports them.

## Export for production (Week 13)

Two paths depending on architecture:

- **Transformer-style (Cosmos-Reason)** → **TensorRT-LLM**. Use the
  TRT-LLM checkpoint converter for the underlying LLM family. FP8 KV
  cache works on Blackwell.
- **Diffusion-style (Cosmos-Predict, Cosmos-Transfer)** → ONNX export
  per UNet/DiT block + VAE, then TensorRT engines per block. Schedule
  with a thin Python loop or a custom Triton ensemble.

Always:
- Build the engine on the *target* hardware (Spark for local; rebuild on
  a SageMaker `ml.p5*` or `ml.g6*` for AWS — `sm_121` engines do not
  transfer).
- Use `--useCudaGraph` for static-shape sub-graphs.
- Quantize to FP8 (Blackwell) if the accuracy delta is acceptable;
  otherwise BF16 / FP16.

## Hand-offs

- Engine perf in TRT/TRT-LLM → `model-deployer` and
  `cuda-perf-profiler`.
- Wiring the inference endpoint into the NextJS DeepAgent →
  `langchain-deepagents-architect`.
- Spatial-CV theory questions → `spatial-intel-researcher`.
