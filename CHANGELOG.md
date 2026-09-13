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
- `python -m engine.hardware` prints the detected profile and tier budget.

### Changed
- Plan revised to "orchestrator first" on top of `kt-kernel` + `sglang-kt`
  (KTransformers archived its YAML injection framework). Minimum VRAM target
  lowered to 6 GB; GGUF via the LLAMAFILE backend is the first weight path.
  Entrypoint renamed to `engine.serve`; the planned custom API server was
  dropped because sglang-kt already serves an OpenAI-compatible API.
