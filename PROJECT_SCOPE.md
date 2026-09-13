# Project Scope

## Goal

Make large sparse Mixture-of-Experts LLMs run on a single consumer machine
(one GPU with 6-24 GB VRAM, system RAM, local NVMe) with zero manual tuning,
and make it obvious — through a Gradio UI — where every expert lives (GPU,
RAM, or disk) and what it costs.

Reference test machine: RTX 3060 Laptop (6 GB VRAM), i5-11400H (6 cores,
AVX-512), 38 GB RAM, NVMe SSD, running Qwen3-30B-A3B at Q4_K_M.

## Approach: integrate tested projects

| Concern | Project used | Fast-MoE's part |
|---|---|---|
| Inference, CPU/GPU expert placement, context sizing | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` with `--fit` | pick defaults, apply UI overrides, supervise |
| RAM ↔ NVMe paging of weights | llama.cpp mmap + OS page cache | measure and visualize (M9) |
| OpenAI-compatible API, metrics | `llama-server` (`/v1/chat/completions`, `/metrics`) | none |
| Model downloads | `huggingface_hub` | quant selection, disk checks |
| UI | Gradio | config, expert map, resources, chat |
| Containers | official `ghcr.io/ggml-org/llama.cpp` server images as the base | one image + compose files |
| Hardware facts | NVML (`nvidia-ml-py`), `psutil` | cgroup-aware profile, tier budgets |

## In scope (v1)

- Zero-config start: one command or `docker compose up`.
- Quantized GGUF models (Q4_K_M default); generic across MoE families
  llama.cpp supports (Qwen3-MoE, Mixtral, DeepSeek, ...).
- Gradio UI: configuration, live expert-placement map, resource usage, chat.
- Docker with NVIDIA GPU passthrough, and a CPU-only compose file.
- Benchmarks against manual `--n-cpu-moe` tuning on the same hardware.

## Explicit non-goals (v1)

- **No training or fine-tuning.** Inference only.
- **No custom kernels or inference engine.** llama.cpp does the compute.
- **No full-precision weights.** Quantized GGUF only.
- **No multi-GPU or multi-node serving.**
- **No dense (non-MoE) models as a target.**

## Later / optional

- KTransformers (`kt-kernel` + `sglang-kt`) backend for AMX-capable, high-RAM
  machines (planner code kept in `engine/ktx_bridge/`).
- Expert prefetching / NVMe pinning experiments, only if M9 measurements show
  the OS page cache falls short.

See [TASKS.md](TASKS.md) for milestones and the log of plan revisions.
