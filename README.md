# Fast-MoE

**Run large Mixture-of-Experts LLMs on a single consumer/workstation GPU by
tiering weights and KV cache across VRAM, system RAM, and local NVMe — with
zero manual memory tuning.**

> Status: early scaffolding (see [TASKS.md](TASKS.md) for milestone progress).
> Not yet functional — this README describes the target design.

## The problem

Sparse MoE models (Qwen3-30B-A3B, DeepSeek-V2/V3, Mixtral, ...) activate only
a handful of experts per token, but ship *all* experts on disk. Running them
today usually means either (a) enough VRAM to hold every expert, which
consumer GPUs don't have, or (b) a static, hand-tuned CPU/GPU split (e.g.
llama.cpp's `--n-cpu-moe N`) that a human has to pick per model, per machine.

Fast-MoE automates that split across three tiers — VRAM (hot), system RAM
(warm), local NVMe (cold) — for both **expert weights** and the **KV cache**,
and picks the split automatically from a hardware profile instead of manual
flags.

## Architecture

Fast-MoE is not a from-scratch inference engine. It builds on
[KTransformers](https://github.com/kvcache-ai/ktransformers) (Tsinghua,
SOSP'25, Apache-2.0), which already provides fast CPU/GPU hybrid MoE
execution with quantized expert kernels. Fast-MoE adds the layer
KTransformers doesn't have:

```
┌─────────────────────────────────────────────────────────────────┐
│                         engine/cli.py                            │
│              zero-flag entrypoint, wires everything below         │
└───────────┬─────────────────────────────────────────┬───────────┘
            │                                         │
┌───────────▼───────────┐                 ┌───────────▼───────────┐
│  engine/hardware/       │                 │  engine/models/         │
│  profiler.py            │──budget plan──▶│  descriptor.py          │
│  allocator.py           │                 │  loader.py, moe_router.py│
└─────────────────────────┘                 └───────────┬───────────┘
                                                          │ injection rules
                                             ┌───────────▼───────────┐
                                             │  engine/ktx_bridge/     │
                                             │  → KTransformers runtime │
                                             └───────────┬───────────┘
                                                          │
┌─────────────────────────────────────────────────────────▼───────┐
│  engine/memory/  paged_cache.py · swap_backend.py · evictor.py    │
│  VRAM (hot) ⇄ RAM (warm) ⇄ NVMe (cold) — KV pages + cold experts   │
└─────────────────────────────────────────────────────────────────┘
            │                                         │
┌───────────▼───────────┐                 ┌───────────▼───────────┐
│  engine/api/server.py   │                 │  ui/app.py (Gradio)     │
│  OpenAI-compatible API  │                 │  config + live memory-  │
└─────────────────────────┘                 │  map visualization      │
                                             └─────────────────────────┘
```

See [PROJECT_SCOPE.md](PROJECT_SCOPE.md) for what's explicitly in and out of
scope, and [TASKS.md](TASKS.md) for the milestone-by-milestone build plan
with the research each milestone is based on.

## Quickstart (target — not yet implemented)

```bash
docker compose up                          # GPU host, auto-detects everything
docker compose --profile cpu up            # CPU-only fallback
```

or natively:

```bash
pip install -e .
python -m engine.cli --model Qwen/Qwen3-30B-A3B
```

Then open `http://localhost:7860` for the Gradio config/visualization UI, or
talk to `http://localhost:8000/v1/chat/completions` directly.

## Benchmarks

Not yet available — tracked in [TASKS.md](TASKS.md) milestone M10. When
published, numbers will compare against the llama.cpp `--n-cpu-moe` baseline
on identical hardware and be reproducible via `benchmarks/bench_throughput.py`.

## Credits / research this project builds on

- [KTransformers](https://github.com/kvcache-ai/ktransformers) — hybrid
  CPU/GPU MoE execution engine this project depends on (SOSP'25).
- [MoE-Infinity](https://arxiv.org/abs/2401.14361) — activation-aware expert
  caching methodology.
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) — paged KV cache
  design this project's memory manager follows.
- [PreScope](https://arxiv.org/abs/2509.23638) — route-aware expert
  prefetching precedent.

## License

Apache-2.0 — see [LICENSE](LICENSE).
