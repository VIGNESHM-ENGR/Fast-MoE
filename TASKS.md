# Tasks

Living, milestone-grouped task list. Check items off as they land. Each
milestone's design rationale and citations live in the approved plan
(`.claude/plans/you-are-an-expert-quiet-reef.md` in this session) and are
summarized in `PROJECT_SCOPE.md`.

## M0 — Repo scaffolding & docs

- [x] `git init`, default branch `main`
- [x] `.gitignore`, `.dockerignore`
- [x] `LICENSE` (Apache-2.0)
- [x] `pyproject.toml` skeleton (deps, `fast-moe` console script)
- [x] Full package/directory skeleton with docstring-only stub modules
- [x] `configs/default_config.yaml`
- [x] `Dockerfile`, `docker-compose.yml` skeletons (real content in M9)
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
- Refinement: the KV-cache vs. hot-expert VRAM split moved to M2 — it needs
  model sizes from the descriptor, so M1 outputs per-tier totals only

## M2 — Generic MoE descriptor + KTransformers bridge

- [ ] Research: diff `config.json` schemas across Qwen3-MoE, Mixtral,
      DeepSeek-V2/V3 (expert-count/top-k field names are not uniform)
- [ ] `engine/models/descriptor.py`: `MoEArchDescriptor` + per-family
      parsers + a generic fallback heuristic for untested architectures
- [ ] Research KTransformers' injection-rule format from its source/docs
- [ ] Model-aware VRAM placement: dense/attention/router weights first,
      then KV cache, then hot experts, within the M1 VRAM budget
- [ ] `engine/ktx_bridge/inject.py`: descriptor + M1 budget → injection
      rule set generator
- [ ] Unit tests against 3 real downloaded `config.json` files
- [ ] Integration test: generated rules load first N layers of each model
      family under KTransformers with no manual edits

## M3 — Quantized weight loading

- [ ] Research KTransformers' supported quantized-expert formats
      (INT4/INT8/GPTQ/FP8) and expected on-disk layout
- [ ] `engine/models/loader.py`: safetensors path for shared/attention
      layers; quantized routed-expert path via KTransformers' kernels
- [ ] `engine/models/loader.py`: GGUF as an alternate source format
- [ ] Verification: forward pass on Qwen3-30B-A3B quantized experts;
      logits/perplexity compared against an HF `transformers` fp16
      reference within an agreed tolerance

## M4 — Paged KV cache manager

- [ ] Re-read vLLM PagedAttention (arXiv:2309.06180) block-table design
- [ ] `engine/memory/paged_cache.py`: fixed-size block allocator, free
      list, per-sequence block table
- [ ] Unit tests (`tests/memory/`): allocate/free/fragmentation behavior
- [ ] Integration test: long sequence forcing block reuse, output-identical
      to a non-paged reference implementation

## M5 — Tiered swap backend + evictor

- [ ] Research task: benchmark `io_uring` Python bindings vs.
      `mmap`+`madvise` vs. a thread-pool executor on the target NVMe;
      pick the backend by measured throughput/latency, not by assumption
- [ ] `engine/memory/swap_backend.py`: VRAM↔RAM via CUDA-stream
      non-blocking copies; RAM↔NVMe via the chosen async I/O backend
- [ ] `engine/memory/evictor.py`: LRU and token-decay eviction policies
- [ ] Microbenchmark: decode throughput doesn't cliff when cold blocks are
      touched, vs. a synchronous-copy baseline

## M6 — Route-aware expert prefetcher

- [ ] Re-read PreScope (arXiv:2509.23638) prefetch-triggering approach
- [ ] `engine/models/moe_router.py`: capture early-layer routing
      decisions, issue async prefetch for predicted downstream experts
- [ ] Benchmark: prefetch hit rate + decode throughput improvement vs. M5
      without prefetching, same hardware

## M7 — OpenAI-compatible API server

- [ ] `engine/api/schemas.py`: OpenAI-compatible request/response models
- [ ] `engine/api/server.py`: `/v1/chat/completions` incl. streaming
- [ ] `engine/cli.py`: `python -m engine.cli --model <hf_path>` wiring
      M1-M6 with zero manual flags
- [ ] Verification: `curl` / `openai` Python client round-trip

## M8 — Gradio UI

- [ ] `ui/state.py`: polling client for the engine's metrics endpoint
- [ ] `ui/panels/config_panel.py`: model picker + VRAM/RAM/NVMe override
      sliders
- [ ] `ui/panels/memory_map_panel.py`: live per-tier expert/KV-block map
      + utilization bars
- [ ] `ui/app.py`: mounts both panels
- [ ] Manual browser walkthrough during a live multi-turn generation

## M9 — Docker packaging

- [ ] Pin an exact KTransformers commit/tag; wire its build into the
      `Dockerfile` GPU builder stage
- [ ] Define a CPU-only install path; complete the `Dockerfile` `cpu` target
- [ ] Verification: `docker compose up` (GPU box) and
      `docker compose --profile cpu up` (CPU box) both reach a working
      API + Gradio UI with no manual flags

## M10 — Benchmarks, README, open-source polish

- [ ] `benchmarks/bench_throughput.py`: Fast-MoE vs. llama.cpp
      `--n-cpu-moe` baseline, same hardware/model
- [ ] Record real numbers in `README.md`
- [ ] `CHANGELOG.md` `v0.1.0` entry
- [ ] Verify README benchmark numbers are reproducible from a clean checkout
