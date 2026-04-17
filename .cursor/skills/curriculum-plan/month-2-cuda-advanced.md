# Month 2 — Advanced CUDA + CV Foundations

Goal: by end of Month 2 you can hand-write a tensor-core GEMM, you've
implemented a fused attention kernel, you understand multi-stream and
basic multi-GPU, and you've connected the GPU world to classical
computer vision (Torralba + Hartley §1-6) with a real GPU-accelerated
image-rectification pipeline.

---

## Week 5 — Tensor cores via WMMA and CUTLASS

**Theme.** You will write your own tensor-core GEMM. You will then read
CUTLASS and understand why it exists.

**Readings.**
- PMPP 4e — Ch 16 (Deep learning, tensor cores).
- CUDA C++ Programming Guide — §B.30 (`wmma`), §B.31 (`mma`).
- Blackwell architecture whitepaper — sections on 5th-gen tensor cores,
  FP4/FP6/FP8 numerics, TMA.
- CUTLASS 3.x docs — *Efficient GEMM in CUDA*, *CuTe* concepts.

**Lab — `labs/week-05-tensor-cores/`.**
1. Write `gemm_wmma_bf16` using `nvcuda::wmma` for M=N=K=4096, BF16 in,
   FP32 accumulate.
2. Write `gemm_mma_bf16` using inline PTX `mma.sync` for the same shape.
3. Build a CUTLASS BF16 GEMM from a template; profile all three.
4. (Stretch) Write an FP8 (`e4m3`) GEMM via CUTLASS targeting Blackwell's
   5th-gen tensor cores.

**Performance target.** Hand-written `gemm_wmma_bf16` ≥ 60% of
`cublasGemmEx(..., CUDA_R_16BF, ...)`.

---

## Week 6 — Fused attention from scratch

**Theme.** Re-implement FlashAttention-style fused attention. This is
the kernel that defines modern transformer perf.

**Readings.**
- *FlashAttention* (Dao et al, 2022) and *FlashAttention-2* (Dao, 2023);
  optionally *FlashAttention-3* (Shah et al, 2024) for Hopper/Blackwell.
- PMPP 4e — Ch 17 (Iterative MRI reconstruction) for the
  iterative/streaming-softmax pattern.
- CUDA C++ Programming Guide — §B.27 (`cuda::pipeline`), §B.28
  (asynchronous barrier).

**Lab — `labs/week-06-fused-attention/`.**
1. Implement a single-precision causal attention kernel (Q, K, V) with
   the streaming-softmax trick, BR×BC tiles, double buffering.
2. Implement a BF16 / FP32-accum variant using WMMA for the QKᵀ and PV
   matmuls.
3. Compare to `torch.nn.functional.scaled_dot_product_attention` (which
   dispatches to the cuDNN/Flash kernels) at sequence lengths
   1k, 4k, 16k, 64k.

**Performance target.** BF16 variant ≥ 60% of PyTorch SDPA.

---

## Week 7 — Multi-stream, async, optional multi-GPU

**Theme.** Real workloads aren't one kernel — they're a graph of kernels
overlapped with copies. Learn streams, events, graphs, and (optionally)
NCCL across two Sparks.

**Readings.**
- CUDA C++ Programming Guide — §3.2.6 (streams), §3.2.8 (graphs),
  §11 (memory pools / `cudaMallocAsync`).
- PMPP 4e — Ch 20 (Programming a heterogeneous computing cluster).
- NCCL developer guide — §1-4.

**Lab — `labs/week-07-streams-graphs/`.**
1. Take your Week-6 attention kernel and your Week-2 GEMM; build a
   3-stream pipeline that overlaps `H→D` copy, attention, GEMM.
2. Capture the pipeline as a `cudaGraph` and re-time vs un-captured.
3. (Stretch — needs a second Spark) Wire NCCL `all_reduce` over CX-7
   between two Sparks; run a 2-rank PyTorch DDP toy model.

**Performance target.** Captured graph delivers ≥ 1.4× speedup vs the
serial version on the same hardware.

---

## Week 8 — Checkpoint: GPU computer-vision pipeline

**Theme.** Bridge to classical CV. Read the foundational chapters of
Torralba and Hartley, then build a GPU pipeline that takes a 4K30 video
stream and outputs rectified, undistorted frames in real time.

**Readings.**
- Torralba/Isola/Freeman — Ch 1 (What is vision?), Ch 2 (Image formation),
  Ch 3 (Filtering), Ch 4 (Edges), Ch 5 (Color), Ch 6 (Texture).
- Hartley & Zisserman — §1 (background), §2 (projective geometry 2D),
  §3 (projective geometry 3D), §6 (camera models — pinhole + lens
  distortion is the spine of this lab).
- Szeliski — §2.1 (geometric primitives), §2.2 (camera model), §2.3
  (photometric image formation).
- NVIDIA Video Codec SDK / NVDEC docs.

**Lab — `labs/week-08-cv-pipeline/`.**
1. Use NVDEC (via PyAV or the Video Codec SDK) to decode a 4K30 H.264
   stream directly into a CUDA buffer.
2. Implement on-GPU:
   - Lens undistortion (Brown-Conrady model, parameters from `cv2.calibrateCamera`).
   - Stereo rectification (if you have a stereo source) or homography-based
     rectification (single source).
   - Color-space conversion (NV12 → RGB).
   - 2D Gaussian filter (separable) as a warm-up FIR.
3. Encode back to H.264 via NVENC OR push frames into a websocket for
   browser preview.

**Performance target.** Sustain ≥ 30 FPS on 3840×2160 on Spark for the
full pipeline.

**Checkpoint rubric.** Strict — needs **17/20**.
