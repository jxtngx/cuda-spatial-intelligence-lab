# Reading Guide — How to Use the Six Textbooks

The six required books cover ~7,000 pages between them.
You will not read all of it, and you should not try to.
This guide tells you what to read, when, and *how* — passive vs active vs reference — so the textbooks become tools, not homework.

> Read this once before Lab 01.
> Skim it again at the start of each month.

---

## Reading modes

Three modes show up in the per-book sections below.
Use them deliberately:

- **R — Read cover-to-cover.** Linear. Take notes. Do exercises. This is rare; reserved for the chapters that *teach* a concept you'll use every lab.
- **A — Read actively, with code open.** Type the code into the lab. Predict the output before running. This is most of your CUDA reading.
- **F — Use as a reference; consult before you write the lab.** Read the relevant section, build a mental map, *then* close the book.
- **S — Skim once for vocabulary, return when you hit the wall.** For chapters whose concepts you'll only need rarely.

---

## 1. PMPP — *Programming Massively Parallel Processors*, 4e

**Authors.** Hwu, Kirk, El Hajj.
**Role.** This is the **CUDA spine** of the course.
Months 1-2 are basically applied PMPP.

### How to read it

PMPP is a textbook in the best tradition: short chapters, end-of-chapter exercises that genuinely build skill.
The 4th edition adds DL-relevant chapters (16 onward) that earlier editions lack.

Read each assigned chapter in two passes:

1. **First pass (45 min).** Read the chapter without code. Note every technical term and put it on an index card / Anki deck.
2. **Second pass (90 min, with editor open).** Type out every code listing into a scratch `.cu` file. Modify one variable; predict the effect; verify with `nsys`/`ncu`.

End-of-chapter exercises: **do all of them** for chapters 4, 5, 6, 10, 11, 16.
Skip exercises in chapters you're using as reference.

### Chapter map → curriculum lab

| Chapters | Mode | Lab(s) |
|---|---|---|
| 1 — Introduction | S | Lab 01 (skim once, never reopen) |
| 2 — Heterogeneous data parallel computing | A | Lab 01 |
| 3 — Multidimensional grids and data | A | Lab 01 |
| 4 — Compute architecture and scheduling | R | Lab 02 (read carefully — the mental model for the rest of CUDA) |
| 5 — Memory architecture and data locality | R | Lab 02 + reread Lab 04 |
| 6 — Performance considerations | R | Lab 03 + reread Lab 04 |
| 7 — Convolution | F | Lab 06 (reference for the attention kernel work) |
| 8 — Stencil | S | Lab 08 (the rectification kernel rhymes with stencils) |
| 9 — Parallel histogram (atomics) | A | Lab 03 |
| 10 — Reduction | R | Lab 03 |
| 11 — Prefix sum (scan) | R | Lab 03 |
| 12 — Merge | F | Lab 07 (consult when async copies / pipelines come up) |
| 13 — Sorting | S | Mainly skim |
| 14 — Sparse matrices | S | Skim |
| 15 — Graph traversal | S | Skim |
| 16 — Deep learning (tensor cores) | R | Lab 05 |
| 17 — Iterative solvers | A | Lab 06 (the streaming-softmax pattern in fused attention) |
| 18-19 — DL serving / inference | F | Lab 13 (reference when building TRT engines) |
| 20 — Programming a heterogeneous cluster | A | Lab 07 (and the optional 2-Spark NCCL stretch) |
| 21+ — Application case studies | F | Skim now; return when a real workload looks similar |

### Companion to PMPP — the CUDA C++ docs

PMPP teaches concepts.
The official CUDA docs teach syntax and constraints.
You will live in two URLs:

- **CUDA C++ Programming Guide** — the spec. Read §3 (programming interface), §B (built-in functions), §C (CUDA dynamic parallelism). Bookmark §B.16 (warp shuffle), §B.18 (cooperative groups), §B.27 (`cuda::pipeline`), §B.30 (`wmma`), §B.31 (`mma`).
- **CUDA C++ Best Practices Guide** — read §9 (memory optimizations) end-to-end before Lab 02. Reread before Lab 04.

When the `cuda-tutor` subagent cites "PMPP §5.4", it means the section in the 4th edition.
Don't accept "PMPP" without a section.

---

## 2. Iglberger — *C++ Software Design*

**Author.** Klaus Iglberger.
**Role.** The book that turns a competent C++ programmer into a competent C++ *designer*.
Most engineers skip the design layer. Don't.

### How to read it

Iglberger is structured around *guidelines* (numbered like Scott Meyers' *Effective C++*).
Read it **R** mode (cover-to-cover) over Labs 01-07, one chapter per weekend.
Each guideline has a "before" and "after" example — type both into a scratch project and convince yourself the "after" is in fact better.

Most useful guidelines for this lab:

- **Guideline 2** — Design for change.
- **Guideline 3-5** — Separate interfaces from implementations.
- **Guideline 15** — Design for the addition of operations (free functions over methods; this is *the* idiom for `Tensor` / `DeviceBuffer` / kernel launchers).
- **Guideline 19-22** — Type erasure and `std::function`-shaped abstractions.
- **Guideline 25-32** — Modern alternatives to inheritance (Visitor, Strategy, Command via `std::variant` / type erasure).

### Chapter map → curriculum lab

| Chapters | Mode | Lab(s) |
|---|---|---|
| 1 — The Art of Software Design | R | Lab 01 |
| 2 — The Art of Building Abstractions | R | Lab 01 |
| 3 — The Purpose of Design Patterns | R | Lab 01 |
| 4 — The Visitor Design Pattern | R | Lab 02 (apply to GEMM kernel-version selection) |
| 5 — The Strategy and Command Patterns | R | Lab 02 (the `LaunchConfig` builder is a Command) |
| 6 — The Adapter, Observer, and CRTP Patterns | R | Lab 03 |
| 7 — The Bridge, Prototype, External Polymorphism | R | Lab 03 |
| 8 — Type Erasure | R | Lab 05 (when you have a polymorphic kernel selector) |
| 9 — The Decorator Pattern | F | Lab 13 (TRT engine wrapping) |
| 10 — The Singleton Pattern | S | Skim — Iglberger argues against; agree |

### When to consult the `cpp20-tutor`

Bring it any "should this be a class or a free function?", "how do I avoid inheritance here?", or "is this idiomatic C++20?" question.
Cite the Iglberger guideline you're trying to apply.

---

## 3. PPP3 — *Programming: Principles and Practice Using C++*, 3e

**Author.** Bjarne Stroustrup.
**Role.** Foundations.
You will not read this book linearly — you have real programming experience.
Use it **F** mode for any C++ language mechanic you're rusty on.

### How to read it

Skim the table of contents during Lab 01.
Mark the chapters you're already fluent in (likely Ch 1-3, 18, 22+).
Read the chapters below carefully.

### Chapter map → "what to actually read"

| Chapters | Mode | Why |
|---|---|---|
| 4 — Computation | F | If pointer arithmetic and `auto` deduction need a refresh |
| 5 — Errors | A | Read the section on `std::expected`-style error returns |
| 9 — Technicalities | F | Header structure, ODR, linkage — consult on first link error |
| 12 — Classes | R | The mental model for value semantics |
| 13 — A display model | S | Skim |
| 14 — Graphics classes | S | Skip |
| 16-17 — GUI | S | Skip |
| 18 — Containers and iterators | A | If your `std::vector`/`std::span` instincts are weak |
| 19 — Vector and free store | R | Read carefully — RAII fundamentals |
| 20 — Containers and iterators (deeper) | F | Reference |
| 21 — Algorithms | A | Standard algorithms / ranges |
| 22 — Streams | F | Reference |
| 23 — Strings, regex, ... | S | Skim |
| 24-26 — Numerics | F | Reference for `<bit>`, `<numbers>`, `<complex>` |

### When to use it vs Iglberger

| Question | Book |
|---|---|
| "How do I do X in C++?" (mechanics) | PPP3 |
| "Should I do X this way or that way?" (design) | Iglberger |

---

## 4. Torralba / Isola / Freeman — *Foundations of Computer Vision*

**Authors.** Antonio Torralba, Phillip Isola, Bill Freeman.
**Role.** Modern CV theory spine.
Bridges classical signal-processing CV to learned models.
The *most current* of the three CV books.

### How to read it

Read **R** mode for the chapters in scope (1-6 + targeted later chapters).
Each chapter has notebook-style exercises — do at least one per chapter and check your work against the published solutions where available.

### Chapter map → curriculum lab

| Chapters | Mode | Lab(s) |
|---|---|---|
| 1 — What is vision? | R | Lab 08 |
| 2 — Image formation | R | Lab 08 (camera model — also covered by Hartley §6) |
| 3 — Filtering | R | Lab 08 (Gaussian filter you'll write in CUDA) |
| 4 — Edges | R | Lab 08 |
| 5 — Color | A | Lab 08 (NV12 → RGB conversion is a color-space exercise) |
| 6 — Texture | A | Lab 08 |
| 7+ — Features, recognition, learned models | F | Reference for Months 3-4 |
| Chapters on neural CV / generative models | F | Cross-reference when reading the Cosmos paper |

### Pair with

- **Szeliski** (#6 below) — for algorithmic depth on the same topics.
- **Hartley** (#5 below) — for the geometry depth Torralba introduces but does not exhaust.

---

## 5. Hartley & Zisserman — *Multiple View Geometry in Computer Vision*, 2e

**Authors.** Hartley, Zisserman.
**Role.** The geometry spine.
Dense, formal, indispensable.
The classical lens through which you'll read the Cosmos and Gaussian-splatting papers.

### How to read it

Hartley is hard.
Read it **R** mode for the chapters in scope, but budget more time per page.
Two passes per chapter:

1. **Skeleton pass.** Read all the section headers and figure captions. Form an outline.
2. **Equations pass.** Read with paper and pen. Re-derive every numbered equation that introduces a new variable.

The exercises are excellent and hard; do them for §11 (the 8-point algorithm) and §12.

### Chapter map → curriculum lab

| Sections | Mode | Lab(s) |
|---|---|---|
| §1 — Introduction | S | Lab 08 (skim for vocabulary) |
| §2 — Projective geometry 2D | A | Lab 08 |
| §3 — Projective geometry 3D | A | Lab 08 |
| §4 — Estimation 2D projective | F | Lab 09 reference |
| §5 — Algorithm evaluation and error analysis | F | Lab 09 reference |
| §6 — Camera models | R | Lab 08 (pinhole + distortion — your rectification lab) |
| §7 — Computation of the camera matrix | F | Lab 10 reference |
| §8 — More single view geometry | S | Skim |
| §9 — Epipolar geometry, the fundamental matrix | R | Lab 09 (the heart of two-view) |
| §10 — 3D reconstruction of cameras and structure | R | Lab 09 |
| §11 — Computation of the fundamental matrix F | R | Lab 09 (8-point algorithm — implement on GPU) |
| §12 — Structure computation | R | Lab 10 |
| §13 — Scene planes and homographies | R | Lab 10 |
| §14 — Affine epipolar geometry | F | Reference |
| §15 — The trifocal tensor | F | Reference (only if you go deep on three-view) |

### When to consult the `spatial-intel-researcher`

Bring it any "what's the modern (learned) replacement for this Hartley chapter?" question.
Expect citations like: "Hartley §11 vs DUSt3R (Wang et al, arXiv:2312.14132)".

---

## 6. Szeliski — *Computer Vision: Algorithms and Applications*, 2e

**Authors.** Richard Szeliski.
**Role.** The breadth reference.
Use **F** mode almost exclusively.

### How to read it

You will read very little of Szeliski cover-to-cover.
Instead:

- Use it as the *first* place to look up an algorithm by name ("structure from motion", "loopy belief propagation", "stereo matching"). Szeliski's coverage is the best single-volume index in the field.
- For each lab in Month 3, read the relevant Szeliski chapter once, *then* go to Hartley for depth.

### Chapter map → curriculum lab

| Chapters | Mode | Lab(s) |
|---|---|---|
| §2 — Image formation | F | Lab 08 (cross-reference with Torralba 2 + Hartley 6) |
| §3 — Image processing | F | Lab 08 reference |
| §4-7 — Features, matching, segmentation | F | Reference |
| §8 — Dense motion estimation | F | Reference |
| §9 — Image stitching | F | Reference |
| §11 — Structure from motion | A | Lab 10 |
| §12 — Stereo correspondence | A | Lab 10 (you'll implement SGBM in CUDA) |
| §13 — 3D reconstruction | R | Lab 12 (sets up Gaussian splatting context) |
| §14 — Image-based rendering | F | Lab 12 reference |
| §15+ — Recognition / generative models | F | Cross-reference for Cosmos work |

---

## How the books interact (one diagram)

```
                     ┌────────────────────┐
                     │   Iglberger (#2)   │
                     │   How to design    │
                     │   any C++ system   │
                     └─────────┬──────────┘
                               │
                               ▼
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  PPP3 (#3)   │────▶│   Modern C++20      │◀────│   PMPP (#1)      │
│  Mechanics   │     │   on the host       │     │   CUDA on the    │
└──────────────┘     │   (host wrappers)   │     │   device         │
                     └─────────┬───────────┘     └─────────┬────────┘
                               │                           │
                               └───────────┬───────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────┐
                       │      Working CUDA + C++ system    │
                       └─────────┬─────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        ▼                        ▼                         ▼
┌──────────────┐        ┌──────────────────┐      ┌──────────────────┐
│ Torralba (#4)│        │  Hartley (#5)    │      │  Szeliski (#6)   │
│ Modern CV    │◀──────▶│  Geometry spine  │◀────▶│  Breadth ref     │
│ theory       │        │                  │      │                  │
└──────┬───────┘        └─────────┬────────┘      └──────────────────┘
       │                          │
       └────────────┬─────────────┘
                    ▼
        ┌─────────────────────────┐
        │  Spatial Intelligence   │
        │  + NVIDIA Cosmos        │
        │  + Gaussian splatting   │
        └─────────────────────────┘
```

---

## A 4-month reading plan in one paragraph

**Month 1.** PMPP 1-6, 9-11, **R** mode. Iglberger 1-7, **R** mode, one chapter per weekend. PPP3 12, 19, **R** mode; rest **F** mode.

**Month 2.** PMPP 16-17, **R** mode. CUDA C++ Programming Guide §B.27 (`pipeline`), §B.30-31 (WMMA / MMA), **R** mode. CUTLASS docs *Efficient GEMM in CUDA*, **A** mode. Then Lab 08: Torralba 1-6 **R**, Hartley 1-3 + 6 **R**, Szeliski 2-3 **F**.

**Month 3.** Hartley 9-13, **R** mode (this is the hard one — protect the time). Szeliski 11-13 **F** mode in parallel. Lab 11: Cosmos technical report **R** mode. Lab 12: Kerbl et al SIGGRAPH 2023 (Gaussian splatting) **R** mode + read the rasterizer's CUDA source.

**Month 4.** Mostly *docs*, not books.
TensorRT Developer Guide §1-4 + §10 (quantization), **R**.
TRT-LLM quick-start + FP8 on Blackwell, **A**.
Triton model-repository + dynamic batching docs, **A**.
SageMaker BYOC docs + Bedrock custom-import docs, **F**.
LangChain DeepAgents docs § *Quickstart*, *Subagents*, *Backends*, *Skills*, **R**.
Next.js 15 App Router + Vercel AI SDK, **A**.

---

## Notes the agents will hold you to

- When the `curriculum-mentor` says "read PMPP §5.4 first", you read PMPP §5.4 first. Coding before reading is a documented anti-pattern in this lab.
- When you cite a chapter in `report/LAB.md`, cite it with a section number. "PMPP" is not a citation. "PMPP 4e §5.4" is.
- Keep an Anki deck (or equivalent) of vocabulary from PMPP and Hartley. Drill it 5 minutes / morning. Names of metrics (`smsp__warps_active`, `Stall LG Throttle`, etc.) and geometric quantities (epipole, essential matrix, trifocal tensor) need to be *fluent*, not look-up-able.
