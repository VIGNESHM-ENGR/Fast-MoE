# Lessons

Each lesson was hit for real while building Fast-MoE: the symptom, the cause, the fix.

## GPU and host

- **NVIDIA GPU vanished on an ASUS laptop.** `nvidia-smi` fails, `lspci` lists only the Intel
  GPU, `journalctl -k` says `NVRM: No NVIDIA GPU found`. Cause: firmware switched the dGPU off;
  `/sys/devices/platform/asus-nb-wmi/dgpu_disable` reads `1`. Fix (user's terminal):
  `echo 0 | sudo tee /sys/devices/platform/asus-nb-wmi/dgpu_disable`, then reboot or pick Hybrid
  mode. Until then Fast-MoE correctly runs CPU-only.
- **GPU queries**: use NVML through `nvidia-ml-py` (the `pynvml` PyPI package is deprecated).
  NVML ignores `CUDA_VISIBLE_DEVICES`, so the profiler filters devices itself.
  `torch.cuda.mem_get_info` would create a CUDA context and eat VRAM just to measure it.
- **psutil ignores container memory limits**; the profiler reads cgroup v2 `memory.max` /
  v1 `memory.limit_in_bytes` and clamps.
- **CUDA toolkit newer than the driver's CUDA version still works** within a major version
  (built with 13.4 on a driver reporting 13.2).
- **Commands needing sudo** cannot run from an agent's non-interactive shell. Hand the user the
  exact lines for their own terminal, with no leading `!`: in bash `! cmd` inverts the exit
  status, so `! apt-get update && apt-get install ...` skips the install.
- **Container stops hit the full timeout** when the app runs as PID 1: PID 1 ignores SIGTERM
  unless it installs a handler, so `docker compose down` waited 30 s and then killed it. Fix:
  `ui/app.py` handles SIGTERM by exiting through `finally: runner.stop()`, and compose sets
  `init: true`. Shutdown with a model running dropped from 30 s to 1.7 s.
- **Testing Ctrl+C from a script**: background jobs of a non-interactive bash ignore SIGINT, so
  `./start.sh &` then `kill -INT` tests nothing. Start it from Python with
  `subprocess.Popen(..., start_new_session=True)` and send `os.killpg(pid, signal.SIGINT)`.
- **NVIDIA Container Toolkit** is required for `docker compose up` with a GPU:
  `apt-get install nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker &&
  systemctl restart docker`. Check with `docker info | grep -i runtimes` (expect `nvidia`).

## llama.cpp

- **Release tags are `bNNNNN` builds**, several per day; the `v0.x` tags are stale. The native
  build pins `b10937`. Docker images exist only for some builds: `server-cuda-b10920` /
  `server-b10920` are pinned; check a tag with `docker manifest inspect`.
- **`--fit` shrinks context first.** With fit on (the default) it drops context to `--fit-ctx`
  (default 4096) before deciding expert placement. Pass a larger `--fit-ctx` to keep context
  and move more experts to RAM.
- **`llama-fit-params` output** is the last stdout line: `-c N -ngl N -ot "regex=CPU,..."`.
  Parse it with `shlex`; a boundary layer can be split (only `ffn_down` on CPU). Takes ~8 s.
- **`-ngl N` offloads the last N layers** (`i_gpu_start = n_layer + 1 - N` in
  `src/llama-model.cpp`), and `-ot` patterns match with `std::regex_search` on tensor names
  (`src/llama-model-loader.cpp`), so Python `re.search` reproduces placement exactly.
- **Token embeddings always stay on the CPU** (`dev_input` is the CPU device).
- **Expert tensor names** are `blk.N.ffn_{gate,up,down,gate_up}_(ch)exps.weight`; some
  architectures (Gemma 4) use fused `gate_up`.
- **Official Docker image quirks**: binaries have no RPATH, so `LD_LIBRARY_PATH` must include
  `/app` or `llama-server` fails from any other working directory; `fit-params` ships as a
  subcommand (`/app/llama fit-params`), wrapped as `/usr/local/bin/llama-fit-params`; Python
  3.12 is present without `venv`/`ensurepip` (install `python3-venv`).
- **`/metrics`** needs `--metrics`; gauges are `llamacpp:prompt_tokens_seconds`,
  `llamacpp:predicted_tokens_seconds`, `llamacpp:requests_processing`.
- **Streaming**: reasoning arrives as `delta.reasoning_content`; the final chunk carries
  `timings` (read from `chunk.model_extra`) and, with `stream_options={"include_usage": True}`,
  `usage`. Prefer these over client-side estimates.
- **Request extensions** go in `extra_body`: `top_k`, `min_p`, `repeat_penalty`, and
  `chat_template_kwargs: {"enable_thinking": bool}` to toggle reasoning.
- **mmap vs `--load-mode none`** (Qwen3-30B-A3B, 6 GB laptop): mmap loads in 10 s at 195
  prompt tok/s; `--load-mode none` loads in 29 s at 235 prompt tok/s; decode speed is the same.
  mmap stays the default because it lets the OS page weights from NVMe.
- **Default slots** are auto (4) sharing one unified KV cache; `-np 1` changed nothing measurable.
- **"RAM is empty" while a model runs** is the mmap page cache: the model shows as buff/cache in
  `free` and as `RssFile` in `/proc/<llama-server pid>/status`, not as used memory. The dashboard
  reads `RssFile`; psutil's `total - available` alone looks empty.
- **CPU pegged, GPU at 5-13 % during decode** is normal for MoE offload: the CPU computes the
  RAM-side experts for every token and the GPU waits. A large context makes it worse: at 64K,
  Gemma 4 fit to `-ngl 25` of 30 layers, so layers 0-5 ran entirely on the CPU, attention too
  (`cpu_attention_layers` in `engine/llama/runner.py`).
- **A container RAM cap does not break mmap**: page cache counts toward `memory.max` but is
  reclaimable, so the kernel re-reads expert pages from disk. Compose drops `mem_limit: 0`,
  which is how `${FAST_MOE_RAM_LIMIT:-0}` means no cap.

## Hugging Face downloads

- **Xet transfers stall** with log lines like "connection struggling". Set
  `HF_HUB_DISABLE_XET=1` to download over plain HTTPS.
- **A frozen `.incomplete` file is not proof of a stall**: Xet writes elsewhere first. Measure
  the downloader's `write_bytes` in `/proc/<pid>/io` before restarting.
- **Interrupted downloads restart from zero** with `huggingface_hub` 1.31 (each attempt gets a
  new temp name). Delete stale `*.incomplete` files under `models/<repo>/.cache/` to reclaim disk.
- **GGUF repos include look-alike files**: `mmproj-*` (vision projector), `mtp-*` and `dflash-*`
  (speculative drafts) match the quant name; the downloader and `local_model()` skip them.
- **Verify a model exists before saying it doesn't**: search
  `https://huggingface.co/api/models?search=<name>`. Qwen3.6-35B-A3B was wrongly assumed
  nonexistent once and replaced with Qwen3-30B-A3B.

## Gradio 6

- `css`, `theme` and `head` are `launch()` arguments, not `gr.Blocks()` arguments.
- Theme fonts must be `gr.themes.Font(...)` objects; plain strings crash at launch.
- Self-hosted font files need `allowed_paths=[assets]` and URLs of the form
  `/gradio_api/file=<absolute path>`.
- A `gr.Group`'s `elem_classes` land on both its outer and inner div; style the card with
  `.parent > .class` or the border and padding appear twice.
- A long-running UI process keeps serving old code. Before testing, `ss -ltnp | grep 7860` and
  restart it, or use another port via `FAST_MOE_UI_PORT`.
- `gradio_client.Client(url).view_api()` lists the app's endpoints for scripted checks.
- **Event data only reaches the first handler** of a chain: `chatbot.retry(fn).then(g)` passes
  `gr.RetryData` to `fn`; in `g` it fails with `TypeError: 'NoneType' object is not subscriptable`.
  Stash what you need in a `gr.State` in the first step.
- **Stopping a streamed reply**: close the OpenAI stream (`stream.close()`); llama-server cancels
  the generation when the client disconnects (slot free in ~0.6 s). Gradio `cancels=` alone leaves
  the HTTP stream open.
- **Don't shadow handler arguments**: `chat(..., thinking)` reused `thinking` for the thought
  message dict, so `bool(dict)` sent reasoning on every time. `tests/test_chat.py` guards it.
- At 390 px Gradio folds tabs into a `…` menu; in Playwright, click the tab at desktop width and
  then resize.
- **`pkill -f` / `pgrep -f` with a pattern that appears in your own command** matches the agent's
  shell and kills it; match on a path the command line does not contain literally.

## Repo and tooling

- **Anchor ignore patterns**: an unanchored `models/` in `.gitignore` silently ignored
  `engine/models/` and `tests/models/`. Use `/models/*` plus `!/models/README.md`, and run
  `git ls-files --others --ignored --exclude-standard` after touching `.gitignore`.
- **Ruff is pinned** (`ruff==0.16.7`) because its default rule set changes between releases;
  CI reads the pin from `pyproject.toml`. `ui/**` allows nested `with` (SIM117) to mirror the
  Gradio layout.
- **UI boundary catches**: handlers in `ui/app.py` catch broad exceptions to show the error in
  the panel; mark them `# noqa: BLE001` with the reason.

## Rejected paths

- **KTransformers / sglang-kt** needs the full BF16 checkpoint for GPU-side weights, which
  conflicts with Q4-only. Its YAML injection framework is archived; `--kt-num-gpu-experts` is
  per MoE layer; `kt run` falls back to an interactive wizard. The planner stays in
  `engine/ktx_bridge/` as a possible future backend.
- **Separate UI and server containers** would need the Docker socket mounted into the UI to
  restart llama-server; one container is simpler and safer.
- **Fabricated dashboard numbers** (for example "96.8% Pruned", fixed GB/s speeds, fake expert
  IDs) were added once by another agent and removed; `tests/test_board.py` now rejects them.
