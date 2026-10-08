# Fast-MoE on a Jetson Orin NX 16 GB

`./start.sh --nx` runs Qwen3.6-35B-A3B (Q4_K_M, 19 GiB) on a Jetson Orin NX 16 GB with about
6 GB of RAM: the small always-used part of the model sits in GPU memory and every expert is
read from the SSD as needed. It runs natively (no Docker) and serves the OpenAI-compatible API
on port 8080, headless.

> **Status: not yet run on a Jetson.** The `--nx` path was tested on an x86 laptop with the
> Jetson check removed. The speed below is a stand-in measured there, not on an Orin.

## How the memory is used

The Orin's CPU and GPU share the same 16 GB. GPU buffers are pinned RAM, which the page cache
can't reuse, so `--nx` keeps the GPU part as small as possible:

| Part | Where | Size |
|---|---|---|
| Attention, router, shared weights of all 40 layers | GPU (pinned RAM) | 2.13 GiB |
| KV cache at 8,192 tokens + compute buffers | GPU (pinned RAM) | about 0.5 GiB |
| 256 experts × 40 layers | memory-mapped from the SSD, CPU computes them | 16.9 GiB on disk |

Measured total GPU memory: 2,713 MiB. Each token uses 8 of the 256 experts per layer (about
0.53 GiB). The kernel keeps recently used experts in whatever RAM is free and reads the rest
from the SSD. There is no hard cap: with 6 GB free, about 3.3 GB caches experts; closing other
programs gives it more and makes it faster.

Under the hood: `llama-fit-params --cpu-moe -c 8192` plans the placement, then
`llama-server --fit off -c 8192 -ngl -1 -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"`
runs exactly that.

## Expected speed

Stand-in measurement: RTX 3060 Laptop, i5-11400H, NVMe, llama-server limited to 3.3 GB of RAM
(a 3,300 MiB cgroup), so experts are re-read from disk the way they would be on the Orin:

| | Result |
|---|---|
| Decode | 0.66–1.94 tokens/s |
| Prompt processing (short prompt) | about 15 s |
| Disk read | about 100 GB for ~130 generated tokens |

With a cache much smaller than the 16.9 GiB of experts, most expert reads go to the SSD. The
Orin's CPU and SSD are likely slower than this laptop's, so expect at most these numbers. This
is fine for batch jobs; for a live voice conversation (LiveKit) it is too slow. More free RAM
is the one thing that helps most.

## Requirements

- JetPack 6 (Ubuntu 22.04, Python 3.10, CUDA 12). JetPack 5 has Python 3.8, which is too old.
- An NVMe SSD for `models/`. A microSD card or eMMC will be much slower.
- About 25 GB free on it (19 GiB model + 5 GiB headroom the downloader checks for).
- Packages: `sudo apt install git python3-venv python3-pip nvidia-jetpack`
  (`nvidia-jetpack` provides `nvcc`; skip it if `/usr/local/cuda/bin/nvcc` exists).
- Optional, for full speed: `sudo nvpmodel -m 0 && sudo jetson_clocks` (maximum power mode).

## Setup

```bash
git clone https://github.com/VIGNESHM-ENGR/Fast-MoE.git
cd Fast-MoE
python3 models/downloader.py   # space to select Qwen3.6-35B-A3B, enter to download (19 GiB)
./start.sh --nx
```

The first `./start.sh --nx`:

1. creates `.venv` and installs Fast-MoE into it;
2. installs CMake 3.24+ into `.venv` if the system one is older (Ubuntu 22.04 ships 3.22);
3. builds the pinned llama.cpp for Orin (`CUDA_ARCH=87`); this takes a while;
4. downloads the model if `models/` doesn't have it yet;
5. starts the server and prints the placement.

Later runs skip finished steps; add `--no-build` to also skip reinstalling the Python package.
Ready when `curl http://127.0.0.1:8080/health` returns `{"status":"ok"}`. Ctrl+C stops it.

## Options

```bash
FAST_MOE_NX_CTX=4096 ./start.sh --nx      # context size (default 8192); smaller frees ~RAM for experts
FAST_MOE_API_HOST=0.0.0.0 ./start.sh --nx -- --api-key "$KEY"   # reachable from other machines
./start.sh --nx -- --alias qwen3.6 --chat-template-kwargs '{"enable_thinking":false}'
```

Everything after `--` goes to `llama-server`. For LiveKit, see [livekit.md](livekit.md): turn
thinking off, set an API key, and read the speed note above first.

## Not supported on the Jetson yet

- **The dashboard** (`python -m ui.app`): it reads GPU memory through NVML, which Jetson
  doesn't support, so it would show no GPU. `--nx` is headless.
- **Docker**: the official llama.cpp CUDA images are amd64 only.
- **`--ram-limit`**: Docker only. Natively, the experts use whatever RAM is free.

## Troubleshooting

| Message | Fix |
|---|---|
| `--nx is for NVIDIA Jetson` | `/etc/nv_tegra_release` is missing: not a Jetson, or not JetPack |
| `Python 3.10+ is required` | upgrade to JetPack 6 |
| `nvcc not found` | `sudo apt install nvidia-jetpack` |
| `.venv has no pip` | `rm -rf .venv && sudo apt install python3-venv python3-pip`, rerun |
| `Port 8080 is already in use` | stop the other server (an earlier `--nx` run?) |
| Very slow tokens | expected with ~6 GB free (see Expected speed); free more RAM or lower `FAST_MOE_NX_CTX` |
