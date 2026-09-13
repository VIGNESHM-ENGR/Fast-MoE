# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- Inference runtime is llama.cpp's `llama-server` (Q4 GGUF only, automatic
  CPU/GPU expert placement with `--fit`). The earlier KTransformers /
  sglang-kt direction needed full-precision checkpoints; its launch planner is
  kept in `engine/ktx_bridge/` for a possible later backend.
- Minimum VRAM target lowered to 6 GB.
- Python dependencies trimmed to profiling, downloads and UI (no torch).

### Fixed
- `.gitignore` no longer hides `engine/models/` and `tests/models/`.
