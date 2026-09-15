# Architecture

## Data flow

```
GGUF file (models/) ──┬─▶ llama-fit-params ──▶ FitResult (-c, -ngl, -ot "...exps=CPU")
                      │         (engine/llama/fit.py)                │
                      └─▶ gguf.GGUFReader ──▶ ModelLayout (tensor names, sizes)
                                (engine/llama/layout.py)              │
                                                                      ▼
                        expert_placement() ──▶ LayerPlacement per layer (gpu_bytes, cpu_bytes)
                                                                      │
             RunPlan (engine/llama/runner.py) ◀────────────────────────┘
                 │                     │
                 ▼                     ▼
   llama-server --fit off <args>    ui/board.py renders tiers, layer cards, matrix
   (engine/llama/server.py)          │
      │ /v1 OpenAI API :8080         │
      │ /metrics, timings, usage ────┘ ui/metrics.py, ui/chat_hud.py
```

`engine/serve.py` (CLI) and `ui/app.py` (dashboard) both drive the same `Runner`.

## Module map

| Path | Owns |
|---|---|
| `engine/hardware/profiler.py` | GPU via NVML, RAM (cgroup-aware), disk kind and speed probe, CPU flags |
| `engine/hardware/allocator.py` | per-tier budgets with reserves; shown on the Hardware tab |
| `engine/models/model_downloader.py` | `huggingface_hub` downloads, quant matching, `local_model()` lookup |
| `engine/models/catalog.py` | tested models: repo, quant, sizes, expert layout, model-card sampling |
| `engine/llama/fit.py` | run and parse `llama-fit-params` |
| `engine/llama/layout.py` | read GGUF tensors; map `-ngl`/`-ot` onto them with llama.cpp's rules |
| `engine/llama/server.py` | find binaries, port check, supervise `llama-server`, health, log tail |
| `engine/llama/runner.py` | `RunSettings` → `plan_run()` → `RunPlan`; `Runner.launch(plan)` |
| `engine/serve.py` | headless launcher that prints the plan then serves |
| `ui/app.py` | Gradio layout, event handlers, process-global panel state, theme |
| `ui/board.py` | Engine tab HTML: explainer cards, memory tiers, GPU layer cards, layer matrix |
| `ui/chat_hud.py` | chat telemetry gauges |
| `ui/studio.py` | system prompt presets, sampling presets, `build_request()`, model card |
| `ui/metrics.py` | NVML/psutil usage and `/metrics` parsing |
| `ui/assets/board.css`, `ui/assets/fonts/` | all styling; self-hosted OFL fonts |
| `Dockerfile`, `docker-compose*.yml` | one image on the official llama.cpp server image |
| `scripts/build_llama_cpp.sh` | native pinned llama.cpp build |

## Shelved and stub code

Leave these unwired unless a task explicitly revives them:

- `engine/ktx_bridge/launch_plan.py` and `engine/models/descriptor.py`: the KTransformers /
  sglang-kt launch planner from before the switch to llama.cpp. Tested, not used by the app.
- `engine/memory/*`, `engine/models/moe_router.py`, `engine/utils/logger.py`,
  `benchmarks/bench_throughput.py`: docstring-only placeholders for TASKS.md milestones M8/M9.

## Invariants

- `ui/app.py` keeps one process-global `panel` dict and one `Runner`: a single local user.
- Preview refuses to fit while a server runs (it would measure VRAM the server holds); Apply
  stops the server, fits, then launches.
- `LlamaServer.url` always uses loopback, even when listening on `0.0.0.0`.
- GGUF header reads are cached by path, size and mtime (`layout_for`), since a read takes seconds.
- Ports bind to localhost; compose publishes `127.0.0.1:7860` and `127.0.0.1:8080` only.
- Project decisions and their reasons are logged in `TASKS.md` under "Plan revisions";
  product intent lives in `PRODUCT.md` and `PROJECT_SCOPE.md`.
