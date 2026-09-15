# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Warning when the chosen context pushes whole layers, attention and KV cache
  included, onto the CPU: a card above the board, red `CPU` chips in the layer
  map, and a line in `python -m engine.serve`'s plan.
- `./start.sh --ram-limit 20g` (or `FAST_MOE_RAM_LIMIT`) caps the container's
  RAM; the dashboard reads the cap as total RAM.
- `benchmarks/bench_context.py`: decode speed, GPU utilisation and placement
  at several context sizes.

### Fixed
- System RAM looked empty while a model ran: the memory-mapped model file sits
  in the page cache, which Linux does not count as used. The RAM meter now shows
  the model file held in RAM (llama-server's `RssFile`) next to used memory, and
  inside a memory-limited container it reports the container's own usage.

## [1.1.0] - 2026-09-15

### Added
- `./start.sh [auto|gpu|cpu] [--no-browser] [--no-build]`: checks Docker and the
  ports, picks GPU or CPU mode, starts the stack, waits for the dashboard, opens
  it in the browser and streams logs; Ctrl+C runs `docker compose down`.
- CI runs ShellCheck on the shell scripts.
- Chat Studio sidebar: a System Prompt panel with presets (Helpful, Concise,
  Coder, Teacher) and a Generation Tuning panel (temperature, top-p, top-k,
  min-p, presence and repeat penalty, max tokens, reasoning on/off) with a
  "Model card" preset that applies each model's official recommended settings.
- Tested-model catalog with a model card under the engine model picker:
  Qwen3.6-35B-A3B, Gemma 4 26B-A4B and Qwen3-30B-A3B, including models not
  downloaded yet.
- Agent skill (`.claude/skills/fast-moe/`) with project rules, setup, change,
  add-a-model, verification and release procedures, and lessons learned;
  `AGENTS.md` and `CLAUDE.md` point every coding agent to it.
- Gemma 4 26B-A4B support, verified on an RTX 3060 Laptop (6 GB): layers 0-1
  on the GPU, layer 2 split, layers 3-29 in RAM, 10.2 tok/s with reasoning.

### Fixed
- `docker compose down` no longer waits out the 30 s timeout: the dashboard
  handles SIGTERM, stops llama-server cleanly, and compose runs an init process.

## [1.0.0] - 2026-09-13

### Added
- Docker: one image on the pinned official llama.cpp server image with the
  control panel inside. `docker compose up` (NVIDIA GPU) or
  `docker compose -f docker-compose.cpu.yml up` (CPU only); ports are
  published on localhost, `./models` is mounted, and the container runs as
  the host user.
- Gradio control panel (`python -m ui.app`): placement preview and start/stop,
  memory-tier and layer views built only from the real fit plan, live
  VRAM/RAM/throughput, chat with llama.cpp-measured speed and context use,
  hardware profile, and server logs.
- Continuous integration (GitHub Actions): ruff lint and pytest on Python
  3.10 and 3.12. Ruff is pinned so local and CI results match.
- `python -m engine.serve`: one-command launcher. Downloads the model if
  needed, runs llama.cpp's `llama-fit-params`, prints which layers' experts
  sit on the GPU, split, or in RAM (with sizes, from GGUF metadata), then
  starts and supervises `llama-server` with exactly those arguments.
- Repository scaffolding: package layout (`engine/`, `ui/`, `configs/`,
  `benchmarks/`, `tests/`), `pyproject.toml`, `Dockerfile` /
  `docker-compose.yml` skeletons, `.gitignore`, `.dockerignore`,
  Apache-2.0 `LICENSE`.
- Project docs: `README.md`, `PROJECT_SCOPE.md`, `TASKS.md`.
- Hardware profiler (`engine/hardware/profiler.py`): NVML GPU detection,
  cgroup-aware RAM detection, cold-tier device classification and
  throughput probe.
- Budget allocator (`engine/hardware/allocator.py`): per-tier VRAM/RAM/disk
  budgets with safety reserves, user caps, and CPU-only fallback.
- `python -m engine.hardware` prints the detected profile and tier budget,
  including CPU instruction sets, NUMA nodes and GPU compute capability.
- MoE architecture descriptor (`engine/models/descriptor.py`) for Qwen3-MoE,
  Mixtral and DeepSeek-V2/V3, with parameter and KV-cache sizing verified
  against published model sizes.
- Model downloader (`python -m engine.models.model_downloader`) for quantized
  GGUF files into `models/`, built on `huggingface_hub`.
- `scripts/build_llama_cpp.sh`: native CUDA build of a pinned llama.cpp release.

### Changed
- README rewritten for engineers and non-technical readers, with dashboard
  screenshots and measured results.
- Inference runtime is llama.cpp's `llama-server` (Q4 GGUF only, automatic
  CPU/GPU expert placement with `--fit`). The earlier KTransformers /
  sglang-kt direction needed full-precision checkpoints; its launch planner is
  kept in `engine/ktx_bridge/` for a possible later backend.
- Minimum VRAM target lowered to 6 GB.
- Default model is Qwen3.6-35B-A3B (`ggml-org/Qwen3.6-35B-A3B-GGUF`, Q4_K_M).
- Python dependencies trimmed to profiling, downloads and UI (no torch).

### Fixed
- The hardware tab no longer shows "nvme" when the drive type is unknown, and
  the Docker image measures the cold tier on the mounted `/models` drive.
- CPU-only placement no longer claims the GPU holds attention and the KV cache.
- `.gitignore` no longer hides `engine/models/` and `tests/models/`.

[1.1.0]: https://github.com/VIGNESHM-ENGR/Fast-MoE/releases/tag/v1.1.0
[1.0.0]: https://github.com/VIGNESHM-ENGR/Fast-MoE/releases/tag/v1.0.0
