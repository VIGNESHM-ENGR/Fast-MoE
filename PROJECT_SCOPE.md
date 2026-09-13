# Project Scope

## Goal

Let a single consumer/workstation machine (one GPU with 8-24GB VRAM, system
RAM, local NVMe) run large sparse Mixture-of-Experts LLMs beyond what fits
in VRAM alone, by automatically tiering expert weights and KV cache across
VRAM (hot) / RAM (warm) / NVMe (cold) — with no manual layer/offload flags —
and by making that tiering visible and configurable through a Gradio UI.

## In scope (v1)

- **Zero-config hardware profiling & allocation**: auto-detect VRAM, RAM,
  and NVMe throughput; derive tier budgets without user-specified flags.
- **Generic MoE architecture support**: read any HF `config.json` and infer
  expert count/top-k/router shape, rather than hardcoding one model family.
  Validated against Qwen3-MoE, Mixtral, and DeepSeek-V2/V3 shapes.
- **Quantized expert weights** (INT4/INT8/GGUF) as the default loading path.
- **Paged KV cache** with block-table allocation and VRAM→RAM→NVMe eviction.
- **Route-aware expert prefetching** using early-layer routing signals.
- **OpenAI-compatible REST API** (`/v1/chat/completions`, streaming).
- **Gradio UI**: memory-limit configuration + a live view of which expert /
  KV block lives in which tier and what it costs.
- **Docker packaging**: NVIDIA GPU passthrough and a CPU-only fallback mode,
  both zero-config via `docker compose up`.

## Explicit non-goals (v1)

- **No training or fine-tuning support.** Inference only.
- **No multi-GPU / tensor-parallel execution in v1.** The hardware profiler
  and allocator target exactly one GPU device; multi-GPU is a future
  milestone, not part of the initial release.
- **No reimplementation of attention/GEMM kernels.** Fast-MoE depends on
  KTransformers for the CPU/GPU execution kernels rather than writing new
  CUDA/AMX kernels; Fast-MoE's own code is the profiling, tiering, paging,
  prefetching, API, and UI layers around that dependency.
- **No support for non-MoE (dense) models as a first-class target.** The
  generic descriptor may happen to load dense models, but they are not a
  design goal or a testing target.
- **No distributed/multi-node serving.** Single-machine only.

## Architecture summary

See [README.md](README.md) for the diagram and [TASKS.md](TASKS.md) for the
milestone breakdown with citations to the research each component is based
on (KTransformers, MoE-Infinity, vLLM PagedAttention, PreScope).

## Base dependency

Fast-MoE depends on `ktransformers` (Apache-2.0) as an installed library,
pinned to a specific commit/tag, rather than vendoring a hard fork of its
source. Fast-MoE's own code lives entirely in `engine/`, `ui/`, `configs/`,
and `benchmarks/`, and talks to KTransformers only through
`engine/ktx_bridge/`.
