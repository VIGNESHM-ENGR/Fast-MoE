# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Chat Studio sidebar: a System Prompt panel with presets (Helpful, Concise,
  Coder, Teacher) and a Generation Tuning panel (temperature, top-p, top-k,
  min-p, presence and repeat penalty, max tokens, reasoning on/off) with a
  "Model card" preset that applies each model's official recommended settings.
- Tested-model catalog with a model card under the engine model picker:
  Qwen3.6-35B-A3B, Gemma 4 26B-A4B and Qwen3-30B-A3B, including models not
  downloaded yet.

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

[1.0.0]: https://github.com/VIGNESHM-ENGR/Fast-MoE/releases/tag/v1.0.0
