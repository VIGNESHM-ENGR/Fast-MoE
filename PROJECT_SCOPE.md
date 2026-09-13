# Project Scope

## Goal

Let a single consumer/workstation machine (one GPU with 6-24 GB VRAM, system
RAM, local NVMe) run large sparse Mixture-of-Experts LLMs beyond what fits
in VRAM alone, by automatically tiering expert weights and KV cache across
VRAM (hot) / RAM (warm) / NVMe (cold) — with no manual layer/offload flags —
and by making that tiering visible and configurable through a Gradio UI.

Reference test machine: RTX 3060 Laptop (6 GB VRAM), i5-11400H (AVX-512,
6 cores), 38 GB RAM, NVMe SSD, running Qwen3-30B-A3B.

## Approach: orchestrator first

The execution engine is [KTransformers](https://github.com/kvcache-ai/ktransformers)'
`kt-kernel` running inside its SGLang fork `sglang-kt`, which already
provides CPU expert kernels, GPU/CPU expert placement, a paged KV cache,
CPU/disk KV offload and an OpenAI-compatible API. Fast-MoE's own work is
what that stack lacks:

1. **Zero-config**: hardware profiling and a model-aware launch plan that
   derives every engine flag, each with a human-readable reason.
2. **Weights**: fetching the right quantized files into the right tier.
3. **Supervision**: one command / one `docker compose up` to run it all.
4. **Visibility**: a Gradio UI for limits and a live map of which expert
   lives in which tier.
5. **Upstream gaps, measured first**: an NVMe cold tier for experts and
   route-aware expert prefetching — built only where benchmarks show
   upstream falls short.

## In scope (v1)

- Zero-config hardware profiling & allocation (VRAM, RAM, NVMe throughput,
  cgroup limits, CPU instruction sets).
- Generic MoE architecture support from `config.json`, validated against
  Qwen3-MoE, Mixtral and DeepSeek-V2/V3.
- Quantized expert weights, GGUF (LLAMAFILE backend) first.
- OpenAI-compatible REST API (`/v1/chat/completions`, streaming).
- Gradio UI: configuration + live tier visualization + chat.
- Docker packaging: NVIDIA GPU passthrough and a CPU-only fallback.
- Benchmarks against llama.cpp `--n-cpu-moe` on the same hardware.

## Explicit non-goals (v1)

- **No training or fine-tuning.** Inference only.
- **No multi-GPU / tensor-parallel execution.** Exactly one GPU device.
- **No new attention/GEMM kernels.** Kernels come from kt-kernel / SGLang.
- **No dense (non-MoE) models as a target.**
- **No distributed / multi-node serving.** Single machine only.

## Base dependency

Fast-MoE depends on `kt-kernel` and `sglang-kt` (Apache-2.0) as pinned,
installed packages rather than a vendored fork. Fast-MoE's code lives in
`engine/`, `ui/`, `configs/` and `benchmarks/`, and touches KTransformers
only through `engine/ktx_bridge/`.

See [TASKS.md](TASKS.md) for milestones and the log of plan revisions.
