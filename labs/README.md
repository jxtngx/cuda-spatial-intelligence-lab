# labs/

One folder per curriculum lab — 16 labs total, paced roughly one per
week over four months. Each lab is self-contained: its own
`CMakeLists.txt`, its own tests, its own benchmarks, its own profile
reports.

```
labs/lab-NN-<slug>/
  LAB.md              # spec + plan of work + writeup (from /start-lab)
  GLOSSARY.md         # new terms only (CUDA / C++20 / CV / Python bindings)
  CMakeLists.txt
  src/                # C++ / CUDA implementation
  tests/              # GoogleTest unit tests; CPU reference + max-error
  bench/              # microbenchmarks (cudaEvent or nvbench)
  python/             # torch.utils.cpp_extension wrapper + pytest (Lab 02+)
  report/             # /lab-report output + Nsight artifacts (.qdrep, .ncu-rep)
```

## Building any lab

```bash
cd labs/lab-NN-<slug>
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench/bench_<name>
```

Every lab targets `sm_121` (Blackwell on DGX Spark) and C++20. Don't
soften either.

## Adding a new lab

`/start-lab N` will scaffold `labs/lab-NN-<slug>/` from
`labs/_template/`. If you want to add ad-hoc work that isn't a curriculum
lab, copy `_template/` manually:

```bash
cp -r labs/_template labs/scratch-<slug>
```
