# labs/

One folder per curriculum week. Each lab is self-contained: its own
`CMakeLists.txt`, its own tests, its own benchmarks, its own profile
reports.

```
labs/week-NN-<slug>/
  LAB.md              # spec + readings + perf target + rubric (from /start-week)
  CMakeLists.txt
  src/                # C++ / CUDA implementation
  tests/              # GoogleTest unit tests; CPU reference + max-error
  bench/              # microbenchmarks (cudaEvent or nvbench)
  report/             # /lab-report output + Nsight artifacts (.qdrep, .ncu-rep)
```

## Building any lab

```bash
cd labs/week-NN-<slug>
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench/bench_<name>
```

Every lab targets `sm_121` (Blackwell on DGX Spark) and C++20. Don't
soften either.

## Adding a new week

`/start-week N` will scaffold `labs/week-NN-<slug>/` from
`labs/_template/`. If you want to add ad-hoc work that isn't a curriculum
week, copy `_template/` manually:

```bash
cp -r labs/_template labs/scratch-<slug>
```
