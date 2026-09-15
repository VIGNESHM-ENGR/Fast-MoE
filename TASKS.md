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
- [x] Exact `--fit` placement: llama.cpp's `llama-fit-params` prints the fitted
      `-c / -ngl / -ot` arguments (documented tool). `--fit` shrinks context
      to `--fit-ctx` (default 4096) *before* placing experts
- [x] Launch-config experiment (Qwen3-30B-A3B, 787-token prompt, 128 tokens):

      | config | load | prompt tok/s | decode tok/s | RAM (anon / mmap) |
      |---|---|---|---|---|
      | default (4 slots) | 10 s | 189 | 17.6 | 0.3 / 17.2 GiB |
      | `-np 1` | 10 s | 195 | 16.5 | 0.3 / 17.2 GiB |
      | `-np 1 --load-mode none` | 29 s | 235 | 16.5 | 0.3 / 0.2 GiB |

      Decision: keep llama.cpp defaults (mmap, auto slots); offer
      `--no-mmap` as an option (+20% prompt speed, 3x slower start, no NVMe paging)

## M5 — `python -m engine.serve` (launcher & supervisor)

- [x] Locate llama.cpp binaries: `$LLAMA_CPP_BIN_DIR`, native build, `PATH`
- [x] Default model `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M, downloaded on first run
- [x] Run `llama-fit-params` and read the GGUF layout (llama.cpp's `gguf`
      package) in parallel; apply `-ngl` / `-ot` with llama.cpp's own rules
      to show which layers' experts are on GPU, split, or in RAM, with sizes
- [x] Start `llama-server` with the fitted arguments and `--fit off` (shown
      placement = actual placement), `--jinja --metrics`, user overrides
      `--ctx`, `--vram-margin-mib`, `--no-mmap`, passthrough after `--`
- [x] Health check, log file under `~/.cache/fast-moe/logs/`, Ctrl+C stops
      the server (verified: 2 s), clear errors for missing binary/model and busy port
- [x] Verified on Qwen3.6-35B-A3B through the UI (see M6)
- [x] Restart with new settings without exiting: `Runner.launch(plan)` after a
      fit made with no server running (a running server would skew free VRAM)
- [ ] Pick the default `--ctx` for Qwen3.6 from measurements (its KV cache is
      ~4x smaller per token)

## M6 — Gradio UI

- [x] `python -m ui.app` on http://127.0.0.1:7860: tabs for Engine & Memory
      Topology, Chat Studio, Hardware & Benchmarks, Engine Logs
- [x] Config panel: model, context length, VRAM margin, mmap → Preview
      Placement (no server needed) or Apply & Start; shows the exact
      `llama-server` command
- [x] Placement view from the real plan: explainer cards, GPU / RAM / disk
      tiers, per-layer GPU cards, full layer matrix, plain-language summary,
      layers moved off the GPU vs. the previous plan
- [x] Resources: VRAM (NVML), RAM (psutil), prompt/generation tok/s from
      `/metrics`
- [x] Chat with reasoning folded into a "Thinking" section; TTFT and phase
      timers, generation speed and context use taken from llama-server's
      own `timings` and `usage`
- [x] Browser walkthrough with Qwen3.6-35B-A3B on the RTX 3060 Laptop:
      layers 0-3 on GPU, 4 split, 5-39 in RAM at 4K context; live in 19 s;
      550-token answer at 18.3 tok/s (measured by llama.cpp); no page errors
- [ ] Per-expert activity (which experts fire): needs routing statistics
      llama.cpp does not expose yet
- [ ] Model download with progress from the UI
- [x] Chat Studio sidebar: system prompt presets and generation tuning with a
      "Model card" preset; verified via llama-server `/slots` that the chosen
      temperature, top-p, top-k and max tokens reach the model
- [x] Tested-model catalog in the model picker; Gemma 4 26B-A4B (UD-Q4_K_M)
      verified on the RTX 3060 Laptop: layers 0-1 GPU, 2 split, 3-29 RAM,
      live 15 s after Apply, 10.2 tok/s, TTFT 3.6 s, no page errors

## M7 — Docker

- [x] One image on the pinned official llama.cpp server image
      (`server-cuda-b10920`, CPU: `server-b10920`) plus Fast-MoE. The panel
      starts and restarts llama-server with fitted arguments, so both live in
      one container; `llama-fit-params` is a wrapper around `llama fit-params`
- [x] `docker-compose.yml` (NVIDIA GPU) and `docker-compose.cpu.yml` (CPU
      only): ports published on 127.0.0.1 only, `./models` mounted at
      `/models`, runs as the host user so downloads aren't root-owned
- [x] Host prerequisite documented: NVIDIA Container Toolkit
- [x] Found and fixed: the base image's binaries have no RPATH, so
      `LD_LIBRARY_PATH` must include `/app` when started from another directory
- [x] GPU stack verified on the RTX 3060 Laptop with Qwen3.6-35B-A3B: same
      placement as native (0-3 GPU, 4 split, 5-39 RAM), live in 19 s, host
      reaches the API on 127.0.0.1:8080, 18.1 tok/s (native: 18.3)
- [x] CPU stack verified: all 40 layers in RAM, 262,144-token context, live
      in 54 s, 3.9 tok/s decode / 7.2 tok/s prompt on the first request
- Not planned: publishing images to a registry. Users build locally with `docker compose up`.

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

**2026-09-13 — M7: one container instead of UI + server containers**
- The panel restarts llama-server with new fitted arguments; controlling a
  second container would mean mounting the Docker socket into the UI. One
  image built on the official llama.cpp image keeps the native code path.

**2026-09-13 — during M5: target model is Qwen3.6-35B-A3B**
- The user asked for Qwen3.6 from the start; the plan wrongly assumed it did
  not exist and substituted Qwen3-30B-A3B. Qwen3.6-35B-A3B (released
  2026-04-15; 40 layers with 30 linear-attention + 10 full-attention layers,
  256 experts, top-8 + shared expert, 262K context) is supported by llama.cpp
  (`qwen35moe`). Default is now `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M (19.0 GiB).
  Qwen3-30B-A3B stays useful as a second test model.
