---
name: langchain-deepagents-nextjs
description: Build the Month-4 NextJS application backed by a LangChain DeepAgent that consumes home cameras/sensors and routes inference to Spark (Triton) or AWS (SageMaker/Bedrock). Use when scaffolding the app, defining tools, wiring SSE streaming, or persisting agent memory.
---

# NextJS + LangChain DeepAgents capstone app

This is the shape the Week 15-16 capstone takes.

## Stack

- **Frontend**: Next.js 15 (App Router), React 19, TypeScript, Tailwind v4,
  shadcn/ui, `@tanstack/react-query`, **Vercel AI SDK** (`ai` package).
- **Agent runtime**: Python service running **`deepagents`** on top of
  LangGraph; FastAPI exposes SSE.
- **Storage**: Postgres + `pgvector` for scene embeddings; LangGraph's
  Postgres checkpointer for cross-thread agent memory.
- **Inference targets**: Triton on Spark (`http://spark.local:8000`) and
  SageMaker / Bedrock endpoints; agent picks via a tool argument.
- **Sensors**: RTSP cameras via `mediamtx` reverse proxy; optional
  RealSense / OAK-D via a Python sidecar.

## Repo layout

```
app/                          # Next.js 15
  app/
    api/agent/invoke/route.ts # POST: SSE proxy to FastAPI
    api/agent/threads/route.ts
    (chat)/page.tsx           # main chat UI
  components/
    chat/
    sensor-pairing/
    scene-viewer/             # uses gsplat WebGL viewer for Week-12 scene
  lib/
    agent-client.ts
    sse.ts
agent/                        # Python FastAPI + deepagents
  app.py
  tools/
    perceive.py
    reason.py
    retrieve.py
    set_target.py
  memory.py
  prompts.py
  pyproject.toml
sensors/
  rtsp/                       # mediamtx config
  realsense/                  # pyrealsense2 sidecar
deploy/
  docker-compose.yml          # spins up everything for local dev on Spark
```

## DeepAgent definition

```python
# agent/app.py
from deepagents import create_deep_agent
from langgraph.checkpoint.postgres import PostgresSaver
from tools.perceive import perceive_scene, list_cameras
from tools.reason import reason_about_scene
from tools.retrieve import query_spatial_index
from tools.set_target import set_compute_target

system_prompt = """You are HomeMind, a spatial-intelligence agent that
perceives the user's home through cameras and sensors, reasons about
scenes, and answers grounded questions with timestamped frame references.

Always:
- Call `perceive_scene` before reasoning about something the user is
  asking about *now*.
- Call `query_spatial_index` for anything in the past.
- Cite the frame_id and timestamp of any observation you reference.
- Respect the user's privacy toggle (`compute_target`); never send
  frames to AWS if the toggle is `local`.
"""

agent = create_deep_agent(
    tools=[perceive_scene, list_cameras, reason_about_scene,
           query_spatial_index, set_compute_target],
    system_prompt=system_prompt,
    subagents=[
        {"name": "vision",
         "description": "Pulls frames and runs Cosmos-Predict.",
         "tools": [perceive_scene, list_cameras]},
        {"name": "reasoner",
         "description": "Answers Q&A about a perceived scene with Cosmos-Reason.",
         "tools": [reason_about_scene, query_spatial_index]},
    ],
    checkpointer=PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]),
)
```

## SSE wiring (NextJS ↔ FastAPI)

`app/api/agent/invoke/route.ts`:

```ts
export async function POST(req: Request) {
  const body = await req.json();
  const upstream = await fetch(`${process.env.AGENT_URL}/invoke`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}
```

FastAPI side returns `text/event-stream` of typed events:

```
event: token        data: {"text": "Looking at the kitchen…"}
event: tool_call    data: {"name": "perceive_scene", "args": {...}}
event: tool_result  data: {"name": "perceive_scene", "ok": true, "frame_ids": [...]}
event: token        data: {"text": "I see the toaster is on."}
event: end          data: {"thread_id": "..."}
```

Render with the Vercel AI SDK's `useChat` for tokens, plus a custom
`tool-trace` panel.

## Sensor patterns

- **IP camera (RTSP)**: run `mediamtx` on Spark, configure a `path` per
  camera; expose **WebRTC** to the browser (low latency preview) and
  **RTSP** to the perception sidecar (keep frames raw).
- **RealSense D455** / **OAK-D**: a Python sidecar in
  `nvcr.io/nvidia/pytorch:<tag>` with `--device=/dev/bus/usb`; publish
  `(rgb, depth, ts)` to a Redis stream. Perception tool consumes from
  Redis.

## Privacy toggle

`set_compute_target("local"|"cloud")` switches both:
- Where inference runs (Triton vs SageMaker).
- Whether frames may leave the LAN (local: never; cloud: per request).

Render the current target prominently in the UI (green = local, amber =
cloud).

## Capstone deliverable (Week 16)

The DeepAgent answers, with grounded references, at least:

1. "What's happening in the kitchen right now?" (perceive)
2. "Did anyone enter the garage in the last hour?" (retrieve + reason)
3. "Show me where the toolbox was at 3 pm yesterday." (retrieve)
4. "What changed in the living room since this morning?" (compare two
   perceived snapshots)
5. "Pan around my couch." (renders Week-12 Gaussian-splatting scene)

Latency budget: ≤ 3 s end-to-end on Spark; ≤ 5 s on AWS.

## Hand-offs

- TRT engine perf for the served models → `model-deployer` and
  `cuda-perf-profiler`.
- Cosmos fine-tune choices → `cosmos-models` skill +
  `spatial-intel-researcher`.
