# Tasks

Living, milestone-grouped task list. Check items off as they land. Scope and
non-goals live in `PROJECT_SCOPE.md`; decisions that changed the plan are
logged at the bottom of this file.

Principle: **integrate tested community projects, don't invent.** Inference is
llama.cpp's `llama-server`; downloads are `huggingface_hub`; UI is Gradio.
Fast-MoE's own code is the glue, the zero-config defaults and the visuals.

## M0 — Repo scaffolding & docs

- [x] `git init`, default branch `main`, `.gitignore`, `.dockerignore`, Apache-2.0 `LICENSE`
- [x] `pyproject.toml`, package skeleton, `configs/default_config.yaml`
- [x] `README.md`, `PROJECT_SCOPE.md`, `CHANGELOG.md`, `TASKS.md`

## M1 — Hardware profiler & budget allocator

- [x] NVML GPU detection (no CUDA context), honors `CUDA_VISIBLE_DEVICES`
- [x] cgroup-aware RAM detection (Docker memory limits respected)
- [x] Cold-tier storage classification + sequential throughput probe
- [x] CPU instruction sets, physical cores, NUMA nodes, GPU compute capability
- [x] Per-tier budgets with reserves, user caps, CPU-only fallback
- [x] Verified on the RTX 3060 Laptop 6 GB dev machine and in Docker (`-m 6g`)

## M2 — Generic MoE descriptor

- [x] `engine/models/descriptor.py`: Qwen3-MoE, Mixtral, DeepSeek-V2/V3;
      dense / expert / KV-cache sizing verified against published parameter counts
- [x] sglang-kt launch planner (`engine/ktx_bridge/`) — **shelved** after the
      switch to llama.cpp; kept for a possible KTransformers backend later
- [ ] Read the same shape from GGUF metadata (llama.cpp's `gguf` Python
      package) so the UI works from the downloaded file alone

## M3 — Model download

- [x] `engine/models/model_downloader.py` on `huggingface_hub`: exact quant
      matching, multi-part GGUF sets, disk-space check, resume, `--list`
- [x] `models/` folder (git-ignored except its README)
- [x] Qwen3-30B-A3B Q4_K_M downloaded on the dev machine (18,556,685,824 bytes,
      matches Hugging Face)

## M4 — First run on the dev machine

- [x] `scripts/build_llama_cpp.sh`: native CUDA build pinned to llama.cpp `b10937`
- [x] Native build succeeds (CUDA 13.4 toolkit on driver 595.84 / CUDA 13.2)
- [x] `llama-server -m models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf --jinja`
      with default `--fit on`: loads in 9.5 s, 4 slots × 4096 ctx, 6 threads,
      expert tensors overridden to CPU (mmap)
- [x] Memory: 4.8 / 6.0 GiB VRAM; server RSS 18.4 GiB of which 18.1 GiB is the
      memory-mapped GGUF (page cache over NVMe) and 0.37 GiB anonymous
- [x] OpenAI Python client round-trip incl. streaming (TTFT 1.7 s)
- [x] First baseline: decode 16.3 tok/s, prompt 25.6 tok/s (26-token prompt)
- [ ] Capture the exact `--fit` placement (per-layer experts on CPU) — not
      printed at default verbosity; try `-lv 4` / `llama-fit-params`
- [ ] Try `--parallel 1` (one user gets the whole context) and
      `--load-mode none` (llama.cpp suggests it beats mmap with CPU experts)

## M5 — `python -m engine.serve` (launcher & supervisor)

- [ ] Locate `llama-server` (native build, `PATH`, or Docker image)
- [ ] Zero-flag launch: `--fit on`, `--metrics`, `--jinja`, host/port; pass
      only what the user overrides (context, VRAM margin, CPU-expert layers)
- [ ] Parse the fit decision from startup logs into a placement map
      (layer → experts on GPU or CPU) for the UI
- [ ] Health check on `/health`, log capture, graceful shutdown, restart on
      config change
- [ ] Clear errors for the known failures (model missing, OOM, port in use)

## M6 — Gradio UI (designed with the impeccable skill)

- [ ] Model panel: pick/download a GGUF with progress
- [ ] Config panel: context length, VRAM margin, RAM cap, CPU threads →
      restart server, show the resulting launch command
- [ ] Expert map: every layer × expert colored by where it lives (GPU / RAM /
      disk via mmap), sized by bytes
- [ ] Resources: VRAM / RAM / disk usage, tok/s and KV usage from `/metrics`
- [ ] Chat panel wired to the local OpenAI endpoint
- [ ] Browser walkthrough during a live generation

## M7 — Docker

- [ ] `docker-compose.yml`: official `ghcr.io/ggml-org/llama.cpp:server-cuda`
      (GPU) or `:server` (CPU profile) + a slim Fast-MoE UI container
- [ ] Host prerequisite documented: NVIDIA Container Toolkit
- [ ] Verification: `docker compose up` and `docker compose --profile cpu up`
      both reach a working API + UI with no manual flags

## M8 — Benchmarks

- [ ] `benchmarks/bench_throughput.py` via the OpenAI API (TTFT, prompt and
      decode tok/s at several context lengths), plus `llama-bench`
- [ ] `--fit` auto placement vs. manual `--n-cpu-moe` sweeps on the same
      hardware; results in `benchmarks/results/`

## M9 — Beyond RAM: NVMe tier (research)

- [ ] llama.cpp memory-maps GGUF files, so the OS page cache already pages
      expert weights between RAM and NVMe. Measure it with a model larger
      than RAM (RSS, page faults, tok/s) before building anything
- [ ] Only if measurements show a real gap: prefetch / pinning experiments
      (MoE-Infinity, PreScope ideas), or the KTransformers backend

## M10 — Release polish

- [ ] README: architecture diagram, benchmark table, UI screenshots
- [ ] `CHANGELOG.md` `v0.1.0`
- [ ] Clean-checkout reproduction of README numbers

---

## Plan revisions

**2026-09-13 — after M1: orchestrator on KTransformers**
- KTransformers archived its YAML injection framework and now ships as
  `kt-kernel` inside the `sglang-kt` fork. Fast-MoE became an orchestrator.
- v1 VRAM floor lowered to 6 GB (the RTX 3060 Laptop dev machine).

**2026-09-13 — during M2: integrate, don't invent**
- kt-kernel's `kt` CLI already has hardware detection, a model analyzer, an
  empirical GPU-expert tuner and downloads; reuse over re-implementation.

**2026-09-13 — after M2: llama.cpp is the v1 runtime**
- User wants Q4 weights only. sglang-kt loads GPU-side weights from a full
  BF16 HF checkpoint (~61 GB) and a GGUF-only KT setup isn't supported
  upstream, so it no longer fits.
- llama.cpp's `llama-server` runs from the GGUF alone, `--fit on` (default)
  already places MoE experts CPU/GPU and sizes context automatically, it
  memory-maps weights (RAM↔NVMe paging for free), serves an OpenAI API with
  `/metrics`, and ships official CUDA and CPU Docker images.
- Fast-MoE = model downloader + launcher/supervisor + Gradio config and
  expert-placement visualization + compose file. KTransformers is an optional
  later backend for AMX / high-RAM machines.
