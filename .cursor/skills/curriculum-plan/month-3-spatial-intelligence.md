# Month 3 — Multi-View Geometry + NVIDIA Cosmos + 3D Reconstruction

Goal: by end of Month 3 you can recover 3D structure from images both
the classical way (8-point on GPU, sparse SfM) and the learned way
(NVIDIA Cosmos foundation models, Gaussian splatting). You have your
own captured scene rendering at ≥ 30 FPS at 1080p.

---

## Week 9 — Two-view geometry, 8-point on GPU

**Theme.** The geometric core of multi-view CV. You implement the
8-point algorithm and RANSAC on the GPU.

**Readings.**
- Hartley & Zisserman — §9 (epipolar geometry and the fundamental
  matrix), §10 (3D reconstruction of cameras and structure), §11
  (computation of the fundamental matrix F — the 8-point algorithm
  lives here).
- Szeliski — §11 (structure from motion).
- *MAGSAC++* (Barath et al) for a modern RANSAC variant.

**Lab — `labs/week-09-eight-point-gpu/`.**
1. CUDA kernel for normalized DLT (data normalization + SVD-friendly
   matrix construction). Use cuSOLVER for the SVD on the device.
2. CUDA kernel for RANSAC scoring (each thread = one minimal-set
   hypothesis; reduce to best inlier count).
3. Validate on a stereo pair from KITTI: recover F, then triangulate.

**Performance target.** End-to-end estimate of F + 95%-confidence RANSAC
in ≤ 5 ms for a 2048-correspondence problem.

---

## Week 10 — Stereo + sparse SfM

**Theme.** Move from two views to many. You implement a basic stereo
matcher in CUDA and connect to COLMAP for sparse SfM ground truth.

**Readings.**
- Hartley & Zisserman — §12 (structure computation), §13 (scene planes
  and homographies).
- Szeliski — §12 (stereo correspondence), §13 (3D reconstruction).
- COLMAP paper (Schönberger & Frahm, 2016).

**Lab — `labs/week-10-stereo-sfm/`.**
1. Capture (or download from KITTI) ~30 images of a static scene.
2. Run COLMAP to get ground-truth poses.
3. Implement a CUDA semi-global block matching (SGBM) stereo kernel for
   any rectified pair; benchmark vs OpenCV's CPU SGBM and Spark's
   `cv2.cuda.StereoSGBM`.
4. Triangulate the densest 100k points and visualize as a `.ply` in a
   browser-side viewer (skeleton provided in `viewer/`).

**Performance target.** SGBM at 1280×720 ≤ 25 ms per pair on Spark.

---

## Week 11 — NVIDIA Cosmos: foundation models for spatial AI

**Theme.** Move from classical to learned. Pull NVIDIA Cosmos, fine-tune
Cosmos-Predict on a small custom dataset, and qualitatively evaluate
Cosmos-Reason on your scenes.

**Readings.**
- NVIDIA Cosmos paper (the Cosmos World Foundation Model technical report).
- Cosmos GitHub README + the relevant model card(s) for Cosmos-Predict
  and Cosmos-Reason.
- TRL fine-tuning docs (you'll use them; spark-friendly).
- NVIDIA Spatial Intelligence Lab publications relevant to current
  research themes (use `spatial-intel-researcher` to surface them).

**Lab — `labs/week-11-cosmos-finetune/`.**
1. Pull the Cosmos NGC container; verify a baseline inference on a
   provided sample.
2. Curate a small (≥ 200 clips) custom dataset from your own captured
   video and/or a public dataset (Ego4D subset, BDD100K, KITTI, etc.).
3. Fine-tune Cosmos-Predict (or the lightest-weight variant supported)
   for 1-3 epochs using TRL on Spark; log to Trackio.
4. Generate qualitative comparisons (baseline vs fine-tuned) on held-out
   clips.

**Performance target.** Fine-tune completes within 24 hrs on a single
Spark, with measurable improvement on a held-out qualitative comparison.

---

## Week 12 — Checkpoint: Gaussian splatting on a captured scene

**Theme.** End-to-end neural reconstruction of a real scene **you**
captured. Train a 3D Gaussian splatting model on it, render novel views.

**Readings.**
- *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
  (Kerbl et al, SIGGRAPH 2023).
- *Mip-Splatting* (Yu et al, CVPR 2024).
- Nerfstudio docs (`gsplat` backend).
- Hartley §10 once more for the camera-model intuition.

**Lab — `labs/week-12-gaussian-splatting/`.**
1. Capture ≥ 200 photos of a real scene in your home/yard. Run COLMAP
   for poses.
2. Train a 3D Gaussian splatting model (use `gsplat` or the official
   Inria implementation; both have CUDA kernels — read them).
3. Render at 1920×1080 and benchmark FPS on Spark.
4. Read the rasterizer's CUDA source and write a 1-page diagnosis of
   the bottleneck — bring `cuda-perf-profiler` in.

**Performance target.** Trained scene renders at ≥ 30 FPS at 1080p on
Spark with PSNR ≥ 25 dB on held-out views.

**Checkpoint rubric.** Strict — needs **17/20**.

**Note.** The trained scene + rendering binary is reused in Week 16 as
one of the things the DeepAgent can call.
