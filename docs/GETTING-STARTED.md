# Getting Started

Before you touch Week 1, prove that your toolchain works. This guide is
the **first 30 minutes** of the curriculum: confirm versions, learn the
two CLI tools you'll live in (`nvidia-smi` and `nvcc`), and compile one
`.cu` file end-to-end.

> **Target setup.** NVIDIA DGX Spark, DGX OS (Ubuntu-based, ARM64),
> CUDA Toolkit 13+, GCC 13+ (or Clang 17+), CMake ≥ 3.28, Ninja, NVIDIA
> Container Runtime for Docker, NGC login. If you're on a different
> CUDA-capable box, most of this still applies — just substitute your
> compute capability for `sm_121`.

---

## 0. The mental model

You will repeatedly answer four questions:

1. **What GPU do I have, and what does it support?** → `nvidia-smi`,
   `nvidia-smi -q`, `deviceQuery`.
2. **What CUDA toolkit am I compiling against?** → `nvcc --version`,
   `which nvcc`, `nvidia-smi` (driver's max supported CUDA).
3. **What C++ compiler am I pairing with `nvcc`?** → `g++ --version`,
   `clang++ --version`, plus a tiny program that prints `__cplusplus`.
4. **Does a real kernel compile and run on my GPU?** → build and run
   the smoke test in §6.

Memorize that order. When *anything* goes wrong later, you re-run those
checks before you debug anything else.

---

## 1. `nvidia-smi` — the GPU's status bar

`nvidia-smi` (NVIDIA System Management Interface) talks to the **driver**,
not the toolkit. It works even when the CUDA toolkit isn't installed.

### 1.1 First run

```bash
nvidia-smi
```

What you must read off the output, top to bottom:

| Field | Why it matters |
|---|---|
| **Driver Version** | Must support your CUDA Toolkit version (see §3). |
| **CUDA Version** (top right) | The **maximum** CUDA runtime this driver supports. *Not* the toolkit you compile with. |
| **GPU name** (e.g. `NVIDIA GB10` or `Grace Blackwell`) | Confirms the hardware. |
| **Memory-Usage** | Total / used VRAM. On Spark, the 128 GB is *unified* — see §1.4. |
| **GPU-Util / Volatile GPU-Util %** | 0% when idle. If a stale process is pinning your GPU, that's where you'll see it. |
| **Processes** (bottom table) | PID, process name, memory used. Kill stragglers with `kill <pid>` before benchmarking. |

If `nvidia-smi` errors with `command not found` or `Driver/library
version mismatch`, stop. Reinstall the driver or reboot before going
further.

### 1.2 The flags you'll actually use

```bash
nvidia-smi                        # one-shot snapshot (default)
nvidia-smi -L                     # list GPUs with UUIDs (useful in scripts)
nvidia-smi -q                     # full structured dump (read once, then forget)
nvidia-smi -q -d MEMORY           # only the memory section
nvidia-smi -q -d CLOCK            # boost / base / SM clocks
nvidia-smi -q -d COMPUTE          # compute mode, MIG state
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu \
           --format=csv           # scriptable, one row per GPU
nvidia-smi dmon                   # live per-second dashboard
nvidia-smi pmon                   # live per-process counters
nvidia-smi -i 0                   # restrict to GPU index 0
nvidia-smi topo -m                # NVLink / PCIe topology matrix (Spark = 1 GPU)
```

For interactive work, the killer combo is two terminals:

```bash
# Terminal 1: launch your program
./my_kernel_bench

# Terminal 2: watch in real time
watch -n 0.5 nvidia-smi          # or: nvidia-smi dmon
```

### 1.3 Things `nvidia-smi` is *not*

- It is not a profiler. For "why is my kernel slow", use Nsight Systems
  (`nsys`) and Nsight Compute (`ncu`). The `cuda-perf-profiler` agent
  drives both.
- It does not tell you the toolkit version. `CUDA Version` in the
  header is the **driver's max supported runtime**, not what you have
  installed for compilation.

### 1.4 Spark-specific notes

- DGX Spark uses **unified memory** between CPU (Grace) and GPU
  (Blackwell). `nvidia-smi`'s "Memory-Usage" reflects what the GPU
  side currently sees — it can change as pages migrate.
- Spark is a **single-GPU** node, so `nvidia-smi topo -m` will be a
  trivial 1×1. Multi-GPU patterns are practiced via two Sparks over
  ConnectX-7 (Month 4 stretch).
- Compute capability is **`sm_121`** (Blackwell, 5th-gen tensor cores,
  FP4/FP6/FP8). `nvidia-smi --query-gpu=compute_cap --format=csv`
  prints `12.1`.

---

## 2. `nvcc` — the CUDA compiler

`nvcc` is the **toolkit's** C++/CUDA compiler driver. It hands C++ to
your host compiler (`gcc`/`clang`) and CUDA device code to its own
backend (`cicc` → PTX → SASS).

### 2.1 Confirm you have it

```bash
which nvcc
nvcc --version
```

You want output like:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 13.0, V13.0.xx
Build cuda_13.0.r13.0/compiler.xxxxxxxx_0
```

Pair this with `nvidia-smi`'s `CUDA Version`:

- `nvcc --version` ≤ `nvidia-smi`'s `CUDA Version` → ✅ you can build
  and run.
- `nvcc --version` >  `nvidia-smi`'s `CUDA Version` → ❌ driver too old.
  Upgrade the driver, don't downgrade the toolkit.

### 2.2 The flags this curriculum uses by default

```bash
nvcc -std=c++20 \
     -arch=sm_121 \
     -O3 \
     --use_fast_math \
     -Xcompiler -Wall,-Wextra \
     -lineinfo \
     foo.cu -o foo
```

What each one does:

| Flag | Why |
|---|---|
| `-std=c++20` | Required. We use concepts, ranges, `<span>`, designated initializers. |
| `-arch=sm_121` | Generate SASS for Blackwell. Without this, `nvcc` may emit only PTX and JIT at runtime, which is slower the first time and may miss tensor-core paths. |
| `-O3` | Standard release optimization. |
| `--use_fast_math` | Fuse muls/adds, allow approximate transcendentals. Trade ULPs for FLOPs. **Disable** in correctness tests; **enable** in benchmarks. |
| `-Xcompiler -Wall,-Wextra` | Pass warning flags through to the host compiler. |
| `-lineinfo` | Embed source line numbers in SASS. Required for Nsight Compute source/SASS view. |

Other useful flags:

```bash
-G                        # full device debug (turns off opts; use with cuda-gdb)
-keep                     # keep .ptx, .cubin, .sass intermediates
-Xptxas -v                # print register / shared / spill usage per kernel
--ptx                     # stop after PTX generation
--cubin                   # stop after SASS generation
-rdc=true                 # relocatable device code (needed for separable compilation)
-Xcompiler -fopenmp       # pass arbitrary flags to the host compiler
```

`-Xptxas -v` is the single most useful flag during kernel tuning. It
prints lines like:

```
ptxas info    : Compiling entry function '_Z4axpyfPKfPfi' for 'sm_121'
ptxas info    : Function properties for _Z4axpyfPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 384 bytes cmem[0]
```

You'll come to read register pressure and spill bytes the way you read
log lines today.

### 2.3 What `nvcc` is *not*

- Not the linker. For multi-translation-unit projects use `nvcc` to
  link, or hand `.o` files to `g++` and link the runtime explicitly
  (`-lcudart -L${CUDA_HOME}/lib64`).
- Not the only frontend. Clang can compile CUDA too. We default to
  `nvcc` for sm_121 support; `cuda-tutor` will tell you when Clang is
  the better choice.

---

## 3. C++ compiler — the host half of every CUDA build

`nvcc` always pairs with a host compiler. On DGX OS that is GCC.

### 3.1 Check the version

```bash
g++ --version
g++ -dumpfullversion
```

Minimum for this curriculum: **GCC 13** (for full C++20: ranges,
`<format>`, concepts, three-way comparison). GCC 14 is fine.

If you prefer Clang:

```bash
clang++ --version
```

Minimum: **Clang 17**.

### 3.2 Confirm C++20 actually works

Save as `cxx_check.cpp`:

```cpp
#include <iostream>
#include <version>
#include <ranges>
#include <span>

int main() {
    std::cout << "__cplusplus = " << __cplusplus << '\n';
    std::cout << "GCC = "
#ifdef __GNUC__
              << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__
#else
              << "n/a"
#endif
              << '\n';
#ifdef __cpp_concepts
    std::cout << "concepts: " << __cpp_concepts << '\n';
#endif
#ifdef __cpp_lib_ranges
    std::cout << "ranges:   " << __cpp_lib_ranges << '\n';
#endif
#ifdef __cpp_lib_span
    std::cout << "span:     " << __cpp_lib_span << '\n';
#endif

    int data[] = {1, 2, 3, 4, 5};
    std::span s{data};
    auto squared = s | std::views::transform([](int x){ return x * x; });
    for (int v : squared) std::cout << v << ' ';
    std::cout << '\n';
}
```

Build and run:

```bash
g++ -std=c++20 -Wall -Wextra cxx_check.cpp -o cxx_check && ./cxx_check
```

Expected output (numbers vary):

```
__cplusplus = 202002
GCC = 13.2.0
concepts: 201907
ranges:   201911
span:     202002
1 4 9 16 25
```

If `__cplusplus` prints `201703` or smaller, you've picked up an old
compiler — re-check `which g++` and `update-alternatives --display gcc`.

### 3.3 Confirm `nvcc` accepts C++20

Save as `nvcc_check.cu`:

```cpp
#include <cstdio>
#include <span>

__global__ void hello(int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) printf("hi from thread %d\n", i);
}

int main() {
    hello<<<1, 8>>>(8);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::span<const int> s; (void)s;
    std::printf("ok, __cplusplus=%ld\n", __cplusplus);
    return 0;
}
```

Build and run:

```bash
nvcc -std=c++20 -arch=sm_121 -O2 nvcc_check.cu -o nvcc_check && ./nvcc_check
```

You should see eight `hi from thread N` lines (in any order) followed by
`ok, __cplusplus=202002`. If you see all 8 print but `__cplusplus` is
wrong, your host-compiler default is older than the `nvcc` driver
expects — pin it explicitly with `nvcc -ccbin /usr/bin/g++-13 ...`.

---

## 4. CMake + Ninja — the build system this curriculum assumes

`nvcc` is fine for one-file demos. Anything bigger — multiple
translation units, GoogleTest, microbenchmarks, separable compilation,
NGC-vs-host portability — wants a real build system. We use **CMake**
(the *meta*-build that generates build files) plus **Ninja** (the
fast build tool that consumes them).

### 4.1 Verify versions

```bash
cmake --version          # need ≥ 3.28 (for first-class CUDA + C++20 + presets v8)
ninja --version          # any recent version (≥ 1.10) is fine
```

Why **3.28** specifically: it's the first release where
`CMAKE_CUDA_ARCHITECTURES "native"` and per-language standard handling
across `CXX` and `CUDA` are both stable, and where C++20 modules
support is mature enough to leave on. If `cmake --version` is older,
upgrade with Kitware's apt repository or `pip install cmake` — do not
fight an old system CMake.

If `which cmake` resolves to two different binaries on PATH (common on
DGX OS where Snap and apt both install one), pin the right one:

```bash
sudo update-alternatives --config cmake
```

### 4.2 The minimal CUDA + C++20 `CMakeLists.txt`

Read this once end-to-end; every lab is a variation on it.

```cmake
cmake_minimum_required(VERSION 3.28)
project(my_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "121")           # Blackwell, sm_121
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)            # gives us CUDA::cudart, CUDA::cublas, ...

add_executable(my_kernel src/main.cu src/host.cpp)
target_link_libraries(my_kernel PRIVATE CUDA::cudart)
target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
```

What each block is doing:

| Lines | Job |
|---|---|
| `cmake_minimum_required` | Pins CMake behavior; enables modern features. |
| `project(... LANGUAGES CXX CUDA)` | Declaring `CUDA` here makes `nvcc` a first-class compiler — `.cu` files compile without custom rules. |
| `CMAKE_CXX_STANDARD 20` block | Tell CMake *and* the underlying compiler to require C++20 with no GNU extensions. Mirror block for `CUDA`. |
| `CMAKE_CUDA_ARCHITECTURES "121"` | Generate SASS for Blackwell. Equivalent to `nvcc -arch=sm_121`. Use `"native"` to autodetect from the box you're on. |
| `CMAKE_CUDA_SEPARABLE_COMPILATION ON` | Allow `__device__` symbols across translation units. Required for non-trivial libraries; harmless overhead for small ones. |
| `CMAKE_BUILD_TYPE Release` default | Without this, single-config generators (Ninja, Make) build with no optimization. |
| `find_package(CUDAToolkit REQUIRED)` | The modern way (CMake ≥ 3.17) to get `CUDA::cudart`, `CUDA::cublas`, `CUDA::cusolver`, etc. as imported targets. |
| `target_compile_options($<$<COMPILE_LANGUAGE:CUDA>:...>)` | Generator expressions: send these flags **only** to the CUDA compiler. The CXX/CUDA split matters because `-Wpedantic` is a host-compiler flag, while `--use_fast_math` is `nvcc`-only. |

### 4.3 The `CudaLab.cmake` helper (what every lab uses)

To avoid copy-pasting that block sixteen times, the curriculum ships
[`cmake/CudaLab.cmake`](../cmake/CudaLab.cmake). Each lab's
`CMakeLists.txt` reduces to:

```cmake
cmake_minimum_required(VERSION 3.28)
project(week_NN_my_lab LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CudaLab.cmake)
cuda_lab_defaults()

add_library(week_NN_lib STATIC src/foo.cu src/bar.cpp)
target_include_directories(week_NN_lib PUBLIC src)
target_link_libraries(week_NN_lib PUBLIC CUDA::cudart)
```

`cuda_lab_defaults()` sets all of:

- `CMAKE_CXX_STANDARD 20` (no extensions, required)
- `CMAKE_CUDA_STANDARD 20` (no extensions, required)
- `CMAKE_CUDA_ARCHITECTURES 121` unless you've overridden it
- `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- `CMAKE_BUILD_TYPE Release` (if not already set)
- `find_package(CUDAToolkit REQUIRED)` so `CUDA::cudart` etc. are
  available
- The standard host warnings (`-Wall -Wextra -Wpedantic`) and the
  standard CUDA flags (`--use_fast_math -lineinfo
  --expt-relaxed-constexpr --expt-extended-lambda`)

If you find yourself wanting to disable `--use_fast_math` for a
correctness-critical lab, override locally:

```cmake
target_compile_options(my_strict_target PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
```

### 4.4 The standard configure → build → test loop

From any lab folder:

```bash
cmake -S . -B build -G Ninja                    # configure
cmake --build build -j                          # build (parallel)
ctest --test-dir build --output-on-failure      # test
```

What each command actually does:

- **`cmake -S . -B build -G Ninja`** — *configure* step. Reads
  `CMakeLists.txt`, finds compilers and packages, writes
  `build/build.ninja`. `-S` is the source dir, `-B` is the build dir
  (we always put it in `build/` and gitignore it). `-G Ninja` picks
  the generator; without it CMake picks "Unix Makefiles" by default.
- **`cmake --build build -j`** — generator-agnostic *build* step.
  `cmake --build` works for Ninja, Make, MSBuild, Xcode — useful in
  scripts. `-j` lets the generator pick parallelism.
- **`ctest --test-dir build --output-on-failure`** — runs every
  `add_test()` target; `--output-on-failure` dumps stdout/stderr only
  for failing tests, which is what you almost always want.

You should be able to reflexively type those three commands without
thinking. If you're typing `make` or `cd build && ninja`, you've
forgotten the abstraction — `cmake --build` is the Right Way.

### 4.5 The CMake commands you'll actually use

```bash
# Configure with explicit compilers (when defaults are wrong)
CC=gcc-13 CXX=g++-13 CUDAHOST_COMPILER=g++-13 \
    cmake -S . -B build -G Ninja

# Switch build type without nuking the build dir
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Verbose build — see the exact nvcc/g++ command lines
cmake --build build -- -v                        # Ninja: -v is "verbose"
cmake --build build --verbose                    # generator-agnostic

# Build a single target
cmake --build build --target test_axpy

# Re-configure after editing CMakeLists.txt (CMake usually does this
# automatically the next time you build, but force it with):
cmake --build build --target rebuild_cache

# Nuke the build dir cleanly (preferred over `rm -rf build` only
# because it's the same command on every OS)
cmake --build build --target clean

# When you change something fundamental (compiler, generator,
# CMAKE_CUDA_ARCHITECTURES), wipe and reconfigure:
rm -rf build && cmake -S . -B build -G Ninja

# Run a specific test, with output, repeated
ctest --test-dir build -R test_axpy --output-on-failure --repeat until-fail:3

# Show what tests exist
ctest --test-dir build -N

# Run tests in parallel
ctest --test-dir build -j 8 --output-on-failure
```

### 4.6 Cache variables you'll toggle

Pass with `-D<NAME>=<value>` on the configure line. The most useful in
this curriculum:

| Variable | Effect |
|---|---|
| `CMAKE_BUILD_TYPE=Debug` | `-O0 -g`, no NDEBUG. Use with `cuda-gdb`. |
| `CMAKE_BUILD_TYPE=RelWithDebInfo` | `-O2 -g`. Default for Nsight Compute reports. |
| `CMAKE_BUILD_TYPE=Release` | `-O3 -DNDEBUG`. Default for benches. |
| `CMAKE_CUDA_ARCHITECTURES="121"` | Override sm_121 (use `"native"` on a non-Spark box). |
| `CMAKE_EXPORT_COMPILE_COMMANDS=ON` | Writes `compile_commands.json` so clangd / Cursor IntelliSense work. **Set this once per lab.** |
| `CMAKE_VERBOSE_MAKEFILE=ON` | Per-build verbosity even without `--verbose`. |
| `CUDAToolkit_ROOT=/usr/local/cuda` | Where to find `nvcc` if not on PATH. |
| `CMAKE_PREFIX_PATH=/opt/foo` | Extra search paths for `find_package`. |

### 4.7 Presets — stop typing the same flags

Once you've been at this for a week, drop a `CMakePresets.json` in the
lab so you can do `cmake --preset=spark-release` instead. Minimal
example:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "spark-release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/spark-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "121",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    {
      "name": "spark-debug",
      "inherits": "spark-release",
      "binaryDir": "${sourceDir}/build/spark-debug",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug" }
    }
  ],
  "buildPresets": [
    { "name": "release", "configurePreset": "spark-release" },
    { "name": "debug",   "configurePreset": "spark-debug"   }
  ],
  "testPresets": [
    {
      "name": "release",
      "configurePreset": "spark-release",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

Then:

```bash
cmake --preset=spark-release
cmake --build --preset=release -j
ctest --preset=release
```

### 4.8 If `cmake` can't find CUDA

```bash
export CUDAToolkit_ROOT=/usr/local/cuda          # or wherever `which nvcc` resolves
export PATH="${CUDAToolkit_ROOT}/bin:${PATH}"
```

Or pass it once on the command line:

```bash
cmake -S . -B build -G Ninja -DCUDAToolkit_ROOT=/usr/local/cuda
```

### 4.9 Make Cursor / clangd see your build

The single most useful one-liner for IDE integration:

```bash
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json compile_commands.json
```

Cursor's C++ extension will pick up `compile_commands.json` from the
lab root and stop complaining about "unknown include path" for CUDA
headers.

---

## 5. NGC containers — the reproducible path

Local installs drift. NGC containers don't.

```bash
docker login nvcr.io                       # use your NGC API key
docker pull nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
docker run --gpus all --rm -it \
    -v $PWD:/work -w /work \
    nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 \
    bash
```

Inside the container, repeat §1-§3:

```bash
nvidia-smi
nvcc --version
g++ --version
```

`dgx-spark-engineer` is the agent to ask for "which NGC tag should I be
using right now?".

---

## 6. The smoke test — compile and run something real

The repo's root `main.cu` is the canonical first build. From the repo
root:

```bash
nvcc -std=c++20 -arch=sm_121 -O3 -lineinfo main.cu -o vector_add
./vector_add
```

Then watch it on the GPU in another terminal:

```bash
watch -n 0.2 nvidia-smi dmon -c 5
```

Once that works, do the same with `labs/week-0-an-even-easier-introduction-to-cuda/`
to confirm the labs build path is wired up.

---

## 7. Cheat sheet — keep this open in a side pane

```bash
# --- hardware / driver ---
nvidia-smi                                # status snapshot
nvidia-smi -L                             # list GPUs + UUIDs
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
           --format=csv,noheader          # scriptable info
nvidia-smi dmon                           # live counters
watch -n 0.5 nvidia-smi                   # poor man's dashboard

# --- toolkit / compilers ---
which nvcc; nvcc --version
which g++;  g++  --version
which cmake; cmake --version

# --- compile a single .cu file ---
nvcc -std=c++20 -arch=sm_121 -O3 -lineinfo \
     -Xptxas -v --use_fast_math \
     foo.cu -o foo

# --- inspect what nvcc made ---
nvcc -std=c++20 -arch=sm_121 --ptx  foo.cu -o foo.ptx
nvcc -std=c++20 -arch=sm_121 --cubin foo.cu -o foo.cubin
cuobjdump --dump-sass foo                 # see the actual SASS

# --- cmake: configure / build / test ---
cmake -S . -B build -G Ninja              # configure
cmake --build build -j                    # build (parallel)
ctest --test-dir build --output-on-failure
cmake --build build --verbose             # see exact nvcc/g++ commands
cmake --build build --target test_axpy    # build one target
cmake --build build --target clean        # clean
rm -rf build && cmake -S . -B build -G Ninja  # nuke + reconfigure

# --- cmake: switch build types ---
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# --- cmake: IDE integration ---
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json compile_commands.json

# --- cmake: with presets ---
cmake --preset=spark-release
cmake --build --preset=release -j
ctest --preset=release

# --- run an NGC container ---
docker run --gpus all --rm -it -v $PWD:/work -w /work \
    nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 bash
```

---

## 8. Troubleshooting

| Symptom | First thing to try |
|---|---|
| `nvidia-smi: command not found` | Driver not installed. Reinstall via DGX Dashboard or `apt install nvidia-driver-XXX`. Reboot. |
| `Failed to initialize NVML: Driver/library version mismatch` | A driver upgrade landed but the kernel module didn't reload. `sudo reboot`. |
| `nvcc: command not found` | CUDA toolkit not in PATH. `export PATH=/usr/local/cuda/bin:$PATH` and re-`source` your shell. |
| `nvcc fatal: Unsupported gpu architecture 'compute_121'` | Toolkit is older than your driver. Upgrade to CUDA Toolkit 13+. |
| `error: identifier "std::span" is undefined` inside `.cu` | Forgot `-std=c++20` or host compiler is < GCC 13. |
| Kernel "runs" but `nvidia-smi` shows no GPU activity | You launched the kernel but never `cudaDeviceSynchronize()`'d before `main` returned. Add a sync + check `cudaGetLastError()`. |
| `ptxas error : Entry function uses too much shared data` | Reduce TILE size, or split kernel. `cuda-perf-profiler` can guide. |
| Builds OK in your shell, fails in CMake | Different `g++`/`nvcc` on the PATH. `cmake --build build -- -v` shows the exact commands. |
| `cmake: command not found` or version too old | Install/upgrade via Kitware apt repo or `pip install cmake`. Need ≥ 3.28. |
| `No CUDA toolset found` / `CMAKE_CUDA_COMPILER not set` | `nvcc` not on PATH at configure time. `export CUDAToolkit_ROOT=/usr/local/cuda` and add `${CUDAToolkit_ROOT}/bin` to PATH, then `rm -rf build` and reconfigure. |
| `Could NOT find CUDAToolkit` | Same fix as above; `find_package(CUDAToolkit REQUIRED)` is searching with a stale PATH. |
| CMake builds with `-O0` even after `-DCMAKE_BUILD_TYPE=Release` | You changed `CMAKE_BUILD_TYPE` after the first configure but didn't re-run `cmake -S . -B build ...`. Re-configure or wipe `build/`. |
| Cursor / clangd shows red squiggles on CUDA headers | Missing `compile_commands.json`. Reconfigure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and symlink it to the lab root (see §4.9). |
| `nvcc fatal: A single input file is required for a non-link phase` from CMake | Spaces or odd characters in your build path. Move the repo somewhere without spaces. |
| Two CMake binaries on PATH disagree on version | `which -a cmake`, then `sudo update-alternatives --config cmake` or remove the older one. |

When stuck, the right escalation is:

1. **`dgx-spark-engineer`** — driver / toolkit / NGC / hardware quirks.
2. **`cuda-tutor`** — language/runtime errors in CUDA C++.
3. **`cpp20-tutor`** — host-side template / concept / linker errors.
4. **`cuda-code-reviewer`** — once it builds, before you call it done.

---

## 9. Done. Now what?

1. Read [`SYLLABUS.md`](./SYLLABUS.md) once.
2. Read [`WORKFLOW.md`](./WORKFLOW.md) — how to actually use the
   slash commands and subagents you're about to depend on.
3. Open [`CURSOR-DOCS.md`](./CURSOR-DOCS.md) and import the 26 docs
   it lists into Cursor's `@Docs` panel (PTX, C++20, NeMo, TensorRT,
   Cosmos, LangChain DeepAgents, …). The agents in this repo expect
   them by exact name. ~20 minutes of crawling, 16 weeks of payoff.
4. Skim [`READING-GUIDE.md`](./READING-GUIDE.md) and decide which book
   sits on your desk for Month 1 (PMPP).
5. In Cursor, run `/start-week 1`. The `curriculum-mentor` will scaffold
   `labs/week-01-hello-cuda/` and queue the agents you'll need.

Welcome to deep water.
