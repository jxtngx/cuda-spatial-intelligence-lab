---
description: Drive the standard Nsight Systems + Nsight Compute profiling workflow for a kernel that just passed correctness tests. Delegates to cuda-perf-profiler.
---

The user invoked `/profile-kernel`. Run the lab's standard profiling
workflow as defined in the `nsight-profiling` skill.

Workflow:

1. Identify the kernel + benchmark binary. Look in the active lab's
   `bench/` folder. If multiple, ask which.
2. Confirm the binary built recently (offer to `cmake --build build` if
   stale).
3. Delegate to the **`cuda-perf-profiler`** subagent with:
   > "Run the standard 4-step Nsight workflow on `<bench binary>` for
   > kernel `<name>`:
   > 1. `compute-sanitizer` (memcheck + racecheck + synccheck).
   > 2. `nsys profile` with NVTX + CUDA traces, save to `report/`.
   > 3. `ncu` with the standard section set + `--import-source on`,
   >    save `report/ncu_<name>_v<N>.ncu-rep`.
   > 4. Read the report in the canonical order and produce the standard
   >    output template (bottleneck, hypothesis, suggested change,
   >    expected effect).
   > Treat the perf target from the active lab's `LAB.md` as the bar."
4. Print the profiler's output verbatim.
5. Recommend whether the user is ready for `/lab-report` (perf hit) or
   another optimization pass (perf below target).
