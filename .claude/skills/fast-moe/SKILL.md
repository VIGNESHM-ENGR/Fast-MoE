---
name: fast-moe
description: Build, run, modify or release the Fast-MoE repo (llama.cpp MoE runner with a Gradio dashboard and Docker). Use when setting it up, running the dashboard or Docker stack, changing engine/ or ui/ code, adding a model, debugging llama-server placement or downloads, or cutting a release.
---

# Fast-MoE

Fast-MoE runs large Mixture-of-Experts GGUF models on one consumer GPU (6 GB and up) by
letting llama.cpp place expert weights across GPU VRAM and system RAM, and shows that
placement in a Gradio dashboard. Fast-MoE is glue: llama.cpp does inference, placement and
the OpenAI API; `huggingface_hub` does downloads; Gradio draws the UI.

## Rules every task follows

1. **Integrate, don't invent.** Before writing code, look for an existing tool in llama.cpp,
   `huggingface_hub` or Gradio that already does it (for example `llama-fit-params` for
   placement). New engine code needs a measurement that proves upstream falls short.
2. **Truthful numbers.** Every figure the UI or README shows comes from the fit plan, the GGUF
   header, NVML, psutil, `/metrics`, or the server's `timings`/`usage`. Unknown values render
   as "—". Tests in `tests/test_board.py` guard this; extend them when you add a view.
3. **Displayed placement = executed placement.** Fit with no server running, then start
   `llama-server --fit off` with exactly the fitted `-c/-ngl/-ot` arguments.
4. **Q4 GGUF only.** Download quantized GGUF files (Q4_K_M by default); never full-precision
   checkpoints.
5. **Verified, not assumed.** A change is done when it ran: tests and lint pass, and anything
   user-visible was exercised for real (server started, browser walkthrough, Docker stack)
   per [references/verify.md](references/verify.md). Check model names and facts on Hugging
   Face or in llama.cpp source before stating them.
6. **Sole-author commits.** Commit as the repo owner with no `Co-Authored-By` or other
   attribution lines. Messages: imperative subject, body explaining why.
7. **Docs travel with the change.** Update `CHANGELOG.md` (`[Unreleased]`), tick or add items
   in `TASKS.md` (log plan changes under "Plan revisions"), and update `README.md` when
   something user-visible or measured changes.
8. **Scope to the request.** Do what the user asked; propose extras instead of doing them
   (an unrequested image rebuild was rejected once). The user's Docker stack is often running
   with a model loaded: say so before stopping it for GPU work.

## Start of a session

Read "Status and hand-off" at the top of `TASKS.md`: what works, the next tasks in priority
order, known nits, and how this user likes to work. Check `git status`, `docker ps` and
`nvidia-smi` before starting servers; port 8080 or the GPU may already be taken.

## Pick the task

### Set up and run

1. Python env: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`. If `ensurepip` is
   missing, create the venv with `--without-pip` and install with `pip --python .venv/bin/python`.
2. Engine: `scripts/build_llama_cpp.sh` (pinned llama.cpp release, CUDA; `GGML_CUDA=OFF` for CPU).
   Binaries land in `third_party/llama.cpp/build/bin/`.
3. Model: `python -m engine.models.model_downloader` (default Qwen3.6-35B-A3B Q4_K_M, 19 GiB),
   or `--repo <hf-repo> --quant Q4_K_M`; `--list` shows available files.
4. Run one of:
   - dashboard: `python -m ui.app` → http://127.0.0.1:7860, Preview Placement, Apply & Start;
   - headless: `python -m engine.serve [--model ...] [--ctx N]`;
   - Docker, one command: `./start.sh [auto|gpu|cpu] [--no-browser] [--no-build] [--ram-limit 20g]`; it waits for the
     dashboard, opens it, streams logs, and Ctrl+C runs `docker compose down`;
   - plain Compose: `docker compose up` (GPU) or `docker compose -f docker-compose.cpu.yml up`.

Done when `curl http://127.0.0.1:8080/health` returns `{"status":"ok"}` and one chat request
returns an answer with `timings`.

### Change the code

1. Read [references/architecture.md](references/architecture.md) for the module map, data flow
   and which directories are shelved stubs.
2. Write or extend a test first (`tests/` mirrors `engine/` and `ui/`; fixtures and mocks, no GPU).
3. Make the change; keep it inside the module that owns the concern.
4. `ruff check engine ui tests` and `pytest` both pass.
5. Verify per [references/verify.md](references/verify.md): UI changes get a browser
   walkthrough, engine changes get a real server start, Docker changes get a stack run.
6. Update docs (rule 7), commit, push; watch CI (`gh run watch`).

Done when CI is green on the pushed commit and the verification you ran is reported with its
numbers.

### Add or support a model

Follow [references/add-a-model.md](references/add-a-model.md).

### Measure speed or pick a default context

`python -m benchmarks.bench_context <gguf> [--ctx 4096 16384 32768 65536]` fits, starts,
generates 256 tokens and prints a table (context, CPU-only layers, experts GPU/RAM, VRAM, GPU
busy, tok/s). It needs the GPU to itself. Put results in `CatalogModel.default_ctx`, README,
TASKS.md and the reference table in [references/verify.md](references/verify.md).

### Debug

Something odd with the GPU, downloads, llama.cpp placement, Docker or Gradio? Read
[references/lessons.md](references/lessons.md) first; most failures seen so far are there
with their fixes.

### Release

1. Working tree clean and CI green on `main`.
2. Bump `version` in `pyproject.toml`; rename `## [Unreleased]` in `CHANGELOG.md` to
   `## [X.Y.Z] - YYYY-MM-DD` and add the `[X.Y.Z]: .../releases/tag/vX.Y.Z` link at the bottom.
3. Commit `Release vX.Y.Z`, push, then `git tag -a vX.Y.Z -m "Fast-MoE vX.Y.Z"` and push the tag.
4. `gh release create vX.Y.Z --verify-tag --title "Fast-MoE vX.Y.Z" --notes-file <notes>` with
   highlights and measured results. Container images are built locally by users; publish none.

Done when `gh api repos/<owner>/Fast-MoE/releases/latest` returns the new tag.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `FAST_MOE_MODELS_DIR` | `./models` | where GGUF files live |
| `LLAMA_CPP_BIN_DIR` | native build dir | where `llama-server` / `llama-fit-params` are found |
| `FAST_MOE_API_HOST` | `127.0.0.1` | llama-server listen address (`0.0.0.0` in Docker) |
| `FAST_MOE_UI_HOST` / `FAST_MOE_UI_PORT` | `127.0.0.1` / `7860` | dashboard address |
| `FAST_MOE_NVME_CACHE_DIR` | auto | cold-tier directory the hardware profiler measures |
| `FAST_MOE_UID` / `FAST_MOE_GID` | `1000` | container user in compose |
| `FAST_MOE_RAM_LIMIT` | unset (no cap) | container RAM cap in compose, e.g. `20g` |
| `HF_HUB_DISABLE_XET` | unset | set to `1` when Hugging Face Xet downloads stall |
| `LLAMA_CPP_TAG` | `b10937` | llama.cpp release `scripts/build_llama_cpp.sh` builds |
