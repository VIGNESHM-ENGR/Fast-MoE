# Tasks

Living, milestone-grouped task list. Check items off as they land. Scope and
non-goals live in `PROJECT_SCOPE.md`; decisions that changed the plan are
logged at the bottom of this file.

## M0 — Repo scaffolding & docs

- [x] `git init`, default branch `main`
- [x] `.gitignore`, `.dockerignore`
- [x] `LICENSE` (Apache-2.0)
- [x] `pyproject.toml` skeleton (deps, `fast-moe` console script)
- [x] Full package/directory skeleton with docstring-only stub modules
- [x] `configs/default_config.yaml`
- [x] `Dockerfile`, `docker-compose.yml` skeletons
- [x] `README.md`, `PROJECT_SCOPE.md`, `CHANGELOG.md`, `TASKS.md`
- [x] First commit (no co-author/attribution line, per project convention)

## M1 — Hardware profiler & budget allocator

- [x] Decide GPU query path → NVML via `nvidia-ml-py` (no CUDA context
      needed; `pynvml` PyPI package is deprecated). Honors `CUDA_VISIBLE_DEVICES`
- [x] `engine/hardware/profiler.py`: GPU count/name/total-free VRAM
- [x] `engine/hardware/profiler.py`: system RAM total/available (`psutil`),
      clamped to the cgroup v1/v2 memory limit so Docker limits are respected
- [x] `engine/hardware/profiler.py`: cold-tier dir resolution, backing device
      classification (partition / LVM / LUKS / btrfs aware), sequential
      read/write probe with page-cache eviction
- [x] `engine/hardware/allocator.py`: VRAM reserve (15%, min 1 GiB), RAM
      headroom, pinned staging buffer, disk headroom, user caps, CPU fallback
- [x] Unit tests (`tests/hardware/`): 31 tests incl. 6GB laptop, 8GB, 24GB
      workstation, multi-GPU, cgroup-limited, and no-GPU profiles
- [x] Manual run on dev machine (RTX 3060 Laptop 6GB / 38GB RAM / NVMe) and
      in Docker with `-m 6g` + bind-mounted `/mnt/nvme_cache`

## M2 — Generic MoE descriptor + model-aware launch plan

Turns "a HF model id + the M1 budget" into a complete, explained set of
`sglang-kt` launch flags. Pure functions, fully unit-testable without a GPU.

- [x] Research: diff `config.json` fields across Qwen3-MoE, Mixtral,
      DeepSeek-V2/V3 (`num_experts` vs `num_local_experts` vs
      `n_routed_experts`, shared experts, MLA vs GQA attention)
- [x] `engine/models/descriptor.py`: `MoEArchDescriptor` + per-family
      parsers; fail loudly on unknown architectures instead of guessing
- [x] Size model: dense params, per-expert params, per-token KV cache bytes
      (GQA and MLA). Verified against published totals: Qwen3-30B-A3B 30.5B /
      3.3B active, Mixtral 46.7B / 12.9B, DeepSeek-V3 671B / 37B
- [x] GGUF bits-per-weight measured from real file sizes (Q4_K_M 4.86)
- [x] CPU instruction sets, physical cores, NUMA nodes, GPU compute
      capability added to the profiler
- [x] `engine/ktx_bridge/launch_plan.py`: dense weights → KV cache → GPU
      experts → extra context; every flag carries a reason. Flag semantics
      checked in sglang-kt 0.7.0.post3 source (`--kt-num-gpu-experts` is
      per MoE layer; `--kt-threadpool-count` defaults to 2; `--kt-method`
      defaults to AMXINT4)
- [x] Unit tests: 4 real `config.json` fixtures × synthetic hardware profiles
      (53 tests total). Laptop result: 2.87 GiB dense + 19.5K-token KV on GPU,
      16.4 GiB Q4_K_M experts in RAM

## M3 — Weight acquisition

- [ ] Download via `huggingface_hub.snapshot_download` (resumable,
      integrity-checked): BF16 checkpoint (~61 GB) + `Qwen/Qwen3-30B-A3B-GGUF`
      Q4_K_M (18.6 GB), with a disk-space check against the M1 budget
- [ ] Progress events for the UI

## M4 — Launcher & supervisor (`python -m engine.serve --model <id>`)

- [ ] Pin `kt-kernel` + `sglang-kt` versions; install path documented
- [ ] `engine/serve.py`: profile → plan → fetch weights → spawn
      `sglang.launch_server`, capture logs, health-check, graceful shutdown
- [ ] Clear error messages for the known failure modes (OOM at load, CUDA
      compute capability < 8.0, missing AVX2)
- [ ] Verification: Qwen3-30B-A3B serving on the RTX 3060 Laptop 6 GB dev
      machine; `openai` Python client round-trip incl. streaming

## M5 — Baseline benchmarks

- [ ] `benchmarks/bench_throughput.py`: prefill tok/s, decode tok/s, TTFT
      at several context lengths via the OpenAI API
- [ ] Same model/quant/hardware under llama.cpp `--n-cpu-moe` as the
      baseline; record both in `benchmarks/results/`
- [ ] Measure what `--kt-max-deferred-experts-per-token` and GPU expert
      count actually buy on 6 GB, to tune the M2 defaults with data

## M6 — Gradio UI (designed with the impeccable skill)

- [ ] Metrics source: NVML + psutil tier usage, SGLang `/metrics`, and
      `--record-kt-gpu-expert-distribution` stats
- [ ] Config panel: model picker, VRAM/RAM/disk caps → re-plan with
      reasons shown → restart server
- [ ] Memory-map view: every layer × expert, colored by tier (GPU / RAM /
      NVMe), sized/heat-mapped by activation frequency
- [ ] Resource view: per-tier utilization, throughput, KV cache usage
- [ ] Chat panel wired to the local OpenAI endpoint
- [ ] Browser walkthrough during a live generation

## M7 — Docker packaging

- [ ] Multi-stage `Dockerfile` (CUDA 12.x devel builder → runtime) with
      pinned `kt-kernel` / `sglang-kt`
- [ ] Research the CPU-only fallback: `sglang-kt` assumes a GPU, so pick a
      CPU runtime (SGLang CPU backend or llama.cpp) and route to it
- [ ] `docker-compose.yml`: GPU passthrough, NVMe + HF cache bind mounts,
      UI and API ports
- [ ] Verification: `docker compose up` on the dev machine and CPU-only
      profile both reach a working API + UI with no manual flags

## M8 — NVMe cold tier for experts (upstream gap)

- [ ] Research: does kt-kernel's LLAMAFILE backend mmap GGUF (letting the
      OS page cache already act as the RAM↔NVMe tier) or copy into RAM?
      Measure RSS and page faults with a model larger than RAM
- [ ] If needed: cap resident expert memory and page cold experts from
      NVMe with an LRU/decay evictor (`engine/memory/`), MoE-Infinity-style
- [ ] Benchmark a model that does not fit in RAM (e.g. Qwen3-235B-A22B
      GGUF) against the M5 baseline

## M9 — Route-aware expert prefetch (research milestone)

- [ ] Measure cross-layer routing predictability on Qwen3-30B-A3B from
      recorded expert distributions (PreScope, arXiv:2509.23638)
- [ ] Prototype prefetch of predicted experts into RAM/GPU, either as a
      `sglang-kt` patch or a direct `KTMoEWrapper` engine loop
- [ ] Keep only if it beats M5/M8 numbers on the same hardware

## M10 — Release polish

- [ ] README: architecture diagram, benchmark table, quickstart, UI screenshots
- [ ] `CHANGELOG.md` `v0.1.0` entry
- [ ] Verify README numbers reproduce from a clean checkout

---

## Plan revisions

**2026-09-13 — after M1**
- KTransformers archived its YAML injection-rule framework; it now ships as
  `kt-kernel` inside the `sglang-kt` SGLang fork, which already provides a
  paged KV cache, CPU/disk KV offload, an OpenAI API and GPU-expert
  placement. Fast-MoE becomes an **orchestrator first** (auto-derived flags,
  weights, supervision, UI, Docker). Own memory/prefetch code is built only
  where upstream lacks it (M8, M9).
- v1 VRAM floor lowered from 8 GB to **6 GB**; the RTX 3060 Laptop dev
  machine is the reference test box.
- First weight path is **GGUF via the LLAMAFILE backend** (no conversion,
  runs on any AVX2 CPU).
- KV-cache vs. hot-expert VRAM split moved from M1 to M2 (needs model sizes).

**2026-09-13 — during M2: integrate, don't invent**
- kt-kernel ships a `kt` CLI (`run`, `model`, `doctor`, `bench`, `chat`,
  `quant`) plus an empirical GPU-expert tuner (`tuna_engine.py`). Fast-MoE
  reuses these instead of writing its own: `kt doctor` for environment
  checks, `huggingface_hub` / `kt model` for downloads, `kt bench` for
  benchmarks, `tuna` as an optional refinement after first launch.
- Fast-MoE keeps only what upstream lacks: non-interactive zero-config flags
  (`kt run` falls back to an interactive wizard, and its non-interactive
  defaults — AMXINT4, mem-fraction 0.98, 40K KV — don't fit a 6 GB laptop),
  the Gradio UI, and Docker packaging.
- Dropped from M3: custom HTTP-range download of non-expert tensors
  (invention). Download the full checkpoint with `huggingface_hub`.
