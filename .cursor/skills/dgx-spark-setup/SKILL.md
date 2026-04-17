---
name: dgx-spark-setup
description: Bootstrap a fresh NVIDIA DGX Spark for this curriculum — verify hardware, log into NGC, pull baseline ARM64 containers, install CUDA-13/CMake/Ninja/Nsight host tooling, and prove the toolchain works with the standard CMake template. Use when starting work on a new Spark.
---

# DGX Spark first-time setup for cuda-spatial-intelligence-lab

Run this end-to-end once on a fresh Spark. Most of it you can copy-paste.

## 1. Verify hardware + drivers

```bash
uname -m                                # expect aarch64
nvidia-smi                              # expect Blackwell GPU, driver loaded
nvcc --version                          # expect CUDA 13.x
docker info | grep -i nvidia            # expect "Default Runtime: nvidia" or runtime listed
```

If `nvidia-smi` doesn't show the Blackwell GPU, stop and consult the DGX
Spark setup docs (`docs.nvidia.com/dgx/dgx-spark/`) — there is no point
proceeding.

## 2. NGC account + Docker login

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key from https://org.ngc.nvidia.com/setup/api-key>
```

## 3. Install host-side dev tooling

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    gcc-13 g++-13 \
    clang-17 \
    cmake \
    ninja-build \
    git git-lfs \
    ccache \
    pkg-config \
    libssl-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130
cmake --version    # expect >= 3.28
ninja --version
```

## 4. Verify Nsight CLI works on ARM64

```bash
nsys --version
ncu --version
compute-sanitizer --version
```

If any of these are missing, install via the CUDA toolkit metapackage or
download Nsight separately for ARM64 from the NVIDIA developer portal.

## 5. Pull baseline containers

```bash
docker pull nvcr.io/nvidia/pytorch:25.10-py3       # check current tag; ARM64 image
docker pull nvcr.io/nvidia/tensorrt:25.10-py3
docker pull nvcr.io/nvidia/tritonserver:25.10-py3
docker pull nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04   # for clean CUDA-only builds
```

(Versions move; pin in the lab's `Dockerfile` when you want
reproducibility.)

Verify a GPU-enabled container:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 nvidia-smi
```

## 6. Verify the standard CMake template builds

From `labs/week-01-hello-cuda/`:

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Smoke-test a kernel run + a Nsight Systems profile:

```bash
nsys profile --output=build/smoke ./build/bench/bench_axpy
nsys stats build/smoke.qdrep | head -50
```

## 7. (Optional) NVIDIA Sync from your laptop

If you'll work from a workstation:
- Install NVIDIA Sync on the workstation.
- Pair with your Spark's hostname (`spark.local`).
- Map the lab repo to a synced folder so saves on the laptop hot-reload
  on the Spark.

## 8. (Optional) Two-Spark setup

- Connect Sparks via ConnectX-7 ports + DAC/QSFP cable.
- Verify NCCL sees the link:
  ```bash
  NCCL_DEBUG=INFO python -c "
  import torch.distributed as dist
  dist.init_process_group('nccl')
  print(dist.get_world_size())
  "
  ```
- For PyTorch DDP across 2 Sparks: `torchrun --nnodes=2 --nproc_per_node=1
  --rdzv_backend=c10d --rdzv_endpoint=<spark0>:29500`.

## 9. Common gotchas

- **x86 image pulled by mistake**: `docker image inspect <img> | grep -i Architecture`
  — must say `arm64` / `aarch64`. Otherwise QEMU emulation drags
  everything into the dirt.
- **Container can't see GPU**: verify the NVIDIA Container Runtime is
  installed (`apt list --installed | grep nvidia-container-toolkit`).
- **`shm` exhaustion** in PyTorch dataloader: run with `--shm-size=32g`
  or `--ipc=host`.
- **Nsight version skew**: build CUDA + Nsight from the same toolkit
  version. Mismatches produce reports that won't open.
