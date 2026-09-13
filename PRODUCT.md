# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Curious local-AI hobbyists with a consumer GPU (gaming laptop or desktop, 6 GB
VRAM and up) who want a large Mixture-of-Experts model running on their own
machine and want to understand what is happening. Many do not know what an
"expert", "layer", "KV cache" or "VRAM margin" is. They open the UI in a
browser on the same computer that runs the model.

## Product Purpose

Fast-MoE runs MoE LLMs (default: Qwen3.6-35B-A3B, Q4_K_M GGUF) on hardware
that cannot hold the whole model in GPU memory, with zero manual tuning, and
makes it obvious where every part of the model lives — GPU, system RAM, or
paged from disk — and what that costs in speed and memory. Success: a
hobbyist gets a working local chat model in one step and can explain, after
looking at the UI, why some of the model is on the GPU and some is not.

## Positioning

The engines (llama.cpp, KTransformers) already place experts across GPU and
CPU; they do it silently behind flags and logs. Fast-MoE's claim is making
that placement visible and adjustable for non-experts, using the real
decision llama.cpp makes (`llama-fit-params` output mapped onto the model's
actual tensors), not an estimate.

## Operating Context

- Runs locally: `python -m engine.serve` (CLI) or the Gradio UI, on the same
  machine as the model; localhost only, no login.
- Inference is llama.cpp `llama-server` (OpenAI-compatible API on port 8080,
  Prometheus `/metrics`). Models are GGUF files in `models/`.
- Starting a model takes ~10–30 s (fit step ~8 s + load); changing settings
  means restarting the server.
- Reference machine: RTX 3060 Laptop 6 GB, 38 GB RAM, NVMe; Qwen3-30B-A3B
  decodes ~16 tok/s there.

## Capabilities and Constraints

- Available now: hardware profile (GPU/VRAM, RAM incl. cgroup limits, disk
  speed, CPU features); model download by Hugging Face repo + quant; fitted
  placement per layer (experts on GPU / split / RAM, with bytes); start,
  stop, restart with settings (minimum context, VRAM margin, mmap on/off,
  extra llama-server flags); chat through the OpenAI endpoint; `/metrics`.
- Terminology: "expert", "layer", "context length", "VRAM", "RAM", "split
  layer" (some of a layer's expert weights on GPU, the rest in RAM).
- Not available yet: live per-expert activity (which experts are used while
  generating). llama.cpp does not expose routing statistics; the expert map
  must be designed so activity can be layered on later.
- Single user, single GPU, single model at a time. No multi-node.

## Brand Commitments

Name: Fast-MoE. Open source (Apache-2.0), public at
https://github.com/VIGNESHM-ENGR/Fast-MoE. No logo or visual identity exists.

## Evidence on Hand

- Real measurements from the reference laptop (TASKS.md M4/M5): load times,
  prompt/decode tok/s, VRAM and RAM split, fitted placement for
  Qwen3-30B-A3B (experts of layers 0–7 on GPU, 8 split, 9–47 in RAM).
- No users, testimonials, benchmarks on other hardware, or screenshots exist;
  do not invent them.

## Product Principles

1. Show the real decision, never an estimate dressed up as fact.
2. Explain in plain language first; exact numbers, flags and commands are
   one step away, never hidden.
3. Every setting shows its consequence (what moves between GPU and RAM, what
   happens to speed and context) before or right after it is applied.
4. Integrate proven tools; the UI is a window onto llama.cpp, not a new engine.
