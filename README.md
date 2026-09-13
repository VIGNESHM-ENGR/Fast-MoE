# Fast-MoE

**Run large Mixture-of-Experts LLMs on a single consumer GPU (6 GB and up)
by tiering weights and KV cache across VRAM, system RAM, and local NVMe —
with zero manual memory tuning, and a UI that shows exactly where every
expert lives.**

> Status: early development. Hardware profiling works today; serving is in
> progress. See [TASKS.md](TASKS.md) for milestone progress.

## The problem

Sparse MoE models (Qwen3-30B-A3B, DeepSeek-V2/V3, Mixtral, ...) activate only
a handful of experts per token, but ship *all* experts. Running them on a
consumer GPU today means hand-tuning a CPU/GPU split — llama.cpp's
`--n-cpu-moe N`, or a dozen `--kt-*` flags for KTransformers — per model,
per machine, with no visibility into what actually landed where.

Fast-MoE picks that split automatically from a hardware profile, explains
every decision, and lets you watch and adjust it live.

## Architecture

Fast-MoE is an orchestrator, not a new kernel library. Execution runs on
[KTransformers](https://github.com/kvcache-ai/ktransformers)' `kt-kernel`
inside its SGLang fork (`sglang-kt`); Fast-MoE owns everything around it.

```
                ┌──────────────────────────────────────────┐
                │  python -m engine.serve --model <hf id>   │
                │  or: docker compose up                    │
                └────────────────────┬─────────────────────┘
                                     │
   ┌─────────────────────┐   ┌───────▼──────────────┐   ┌─────────────────────┐
   │ engine/hardware/     │   │ engine/models/        │   │ engine/ktx_bridge/   │
   │ VRAM · RAM · cgroup  │──▶│ MoE descriptor from   │──▶│ launch plan: every   │
   │ NVMe speed · CPU ISA │   │ config.json + sizing  │   │ --kt-* flag + reason │
   └─────────────────────┘   └──────────────────────┘   └──────────┬──────────┘
                                                                     │ spawns + supervises
                ┌────────────────────────────────────────────────────▼──────────┐
                │ sglang-kt + kt-kernel                                          │
                │  GPU (hot):  attention · embeddings · KV cache · hot experts   │
                │  RAM (warm): cold experts (GGUF, CPU kernels)                  │
                │  NVMe (cold): weights + overflow tier (M8)                     │
                └──────────────┬─────────────────────────────────┬───────────────┘
                               │ OpenAI API :8000                │ metrics + expert stats
                     ┌─────────▼─────────┐             ┌─────────▼──────────────┐
                     │ your apps / chat  │             │ ui/ (Gradio) :7860      │
                     └───────────────────┘             │ limits · live tier map  │
                                                       └────────────────────────┘
```

See [PROJECT_SCOPE.md](PROJECT_SCOPE.md) for what is in and out of scope.

## Try it now: profile your hardware

```bash
pip install -e .
python -m engine.hardware
```

Example output on the reference laptop (RTX 3060 Laptop 6 GB, 38 GB RAM):

```
== Tier budget ==
VRAM (hot):  4.66 GiB on NVIDIA GeForce RTX 3060 Laptop GPU, 1.00 GiB reserved
RAM (warm):  27.49 GiB  (pinned 2.75 GiB)
Disk (cold): 253.96 GiB at /home/you/.cache/fast-moe/swap
```

## Quickstart (target — not yet implemented)

```bash
docker compose up                          # GPU host, auto-detects everything
docker compose --profile cpu up            # CPU-only fallback
```

or natively:

```bash
python -m engine.serve --model Qwen/Qwen3-30B-A3B
```

Then open `http://localhost:7860` for the UI, or talk to
`http://localhost:8000/v1/chat/completions` directly.

## Benchmarks

Not yet available (TASKS.md M5). Numbers will compare against llama.cpp
`--n-cpu-moe` on identical hardware and be reproducible via
`benchmarks/bench_throughput.py`.

## Credits / research this project builds on

- [KTransformers](https://github.com/kvcache-ai/ktransformers) (SOSP'25) —
  `kt-kernel` CPU/GPU hybrid MoE execution this project runs on.
- [SGLang](https://github.com/sgl-project/sglang) — serving engine, paged KV
  cache, OpenAI API.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361) — activation-aware expert
  caching and offloading.
- [PreScope](https://arxiv.org/abs/2509.23638) — expert prefetching for
  resource-constrained MoE inference.
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) — paged KV cache
  design.

## License

Apache-2.0 — see [LICENSE](LICENSE).
