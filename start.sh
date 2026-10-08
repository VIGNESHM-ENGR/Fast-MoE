#!/usr/bin/env bash
# Start Fast-MoE in Docker, open the dashboard, and stop everything on Ctrl+C.
#
#   ./start.sh              auto: GPU if Docker can use an NVIDIA GPU, otherwise CPU
#   ./start.sh gpu          NVIDIA GPU (docker-compose.yml)
#   ./start.sh cpu          CPU only (docker-compose.cpu.yml)
#   ./start.sh --no-browser --no-build --ram-limit 20g
#   ./start.sh --nx [-- llama-server args]   Jetson Orin NX 16 GB, native (docs/jetson-orin-nx.md)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_URL="http://127.0.0.1:7860"
API_URL="http://127.0.0.1:8080/v1"
WAIT_SECONDS="${FAST_MOE_START_TIMEOUT:-600}"

mode="auto"
open_browser=1
build=1
nx=0
llama_args=()

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  auto | gpu | cpu   device mode (default: auto)
  --no-browser       don't open the dashboard in a browser
  --no-build         start the existing image without rebuilding
  --ram-limit SIZE   cap the container's RAM, e.g. 20g or 16384m (default: no cap)
  --nx               Jetson Orin NX 16 GB: no Docker; set up .venv, build llama.cpp
                     for Orin once, and serve the API with every expert memory-mapped
                     from the SSD (headless, no dashboard)
  -- ARGS            with --nx: pass ARGS to llama-server (e.g. --api-key KEY)
  -h, --help         show this help

Environment:
  FAST_MOE_START_TIMEOUT  seconds to wait for the dashboard (default 600)
  FAST_MOE_RAM_LIMIT      same as --ram-limit
  FAST_MOE_NX_CTX         context with --nx (default 8192)
  FAST_MOE_API_HOST       with --nx: 0.0.0.0 to reach the API from other machines
EOF
}

say() { printf '\033[1;36m[fast-moe]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fast-moe]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[fast-moe]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    auto|gpu|cpu) mode="$1" ;;
    --no-browser) open_browser=0 ;;
    --no-build) build=0 ;;
    --ram-limit)
      [[ $# -ge 2 ]] || die "--ram-limit needs a size, e.g. --ram-limit 20g"
      FAST_MOE_RAM_LIMIT="$2"
      shift
      ;;
    --ram-limit=*) FAST_MOE_RAM_LIMIT="${1#*=}" ;;
    --nx) nx=1 ;;
    --) shift; llama_args=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

port_busy() {
  if command -v ss >/dev/null; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}

# Jetson Orin NX 16 GB: CPU and GPU share 16 GB, so GPU buffers are pinned RAM the page cache
# can't reuse. Keep only attention/shared weights and the KV cache on the GPU (~2.5 GiB for
# Qwen3.6) and leave every expert memory-mapped: the kernel pages them in from the SSD into
# whatever RAM is free. The official llama.cpp CUDA images are amd64-only, hence native.
run_nx() {
  [[ -f /etc/nv_tegra_release ]] || die "--nx is for NVIDIA Jetson (no /etc/nv_tegra_release here)."
  [[ -z "${FAST_MOE_RAM_LIMIT:-}" ]] || die "--ram-limit applies to Docker only; see docs/jetson-orin-nx.md."
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "Python 3.10+ is required (JetPack 6). This is $(python3 --version 2>&1)."
  cd "$ROOT"
  export PATH="$ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"

  if [[ ! -x .venv/bin/python ]]; then
    say "Creating .venv ..."
    python3 -m venv .venv || die "python3 -m venv failed; install it: sudo apt install python3-venv python3-pip"
  fi
  if [[ $build -eq 1 ]] || ! .venv/bin/python -c 'import importlib.metadata as m; m.version("fast-moe")' 2>/dev/null; then
    .venv/bin/python -m pip --version >/dev/null 2>&1 || die ".venv has no pip; remove .venv and rerun, or: sudo apt install python3-pip"
    say "Installing Fast-MoE into .venv ..."
    .venv/bin/python -m pip install --quiet -e . || die "pip install failed (output above)."
  fi

  if [[ ! -x third_party/llama.cpp/build/bin/llama-server || ! -x third_party/llama.cpp/build/bin/llama-fit-params ]]; then
    command -v nvcc >/dev/null || die "nvcc not found; install the CUDA toolkit: sudo apt install nvidia-jetpack"
    # build_llama_cpp.sh needs CMake 3.24+; Ubuntu 22.04 (JetPack 6) ships 3.22.
    if ! cmake --version 2>/dev/null | awk 'NR==1 {split($3, v, "."); exit !(v[1] > 3 || (v[1] == 3 && v[2] >= 24))}'; then
      say "Installing a newer CMake into .venv ..."
      .venv/bin/python -m pip install --quiet "cmake>=3.24" || die "Could not install CMake."
    fi
    say "Building llama.cpp for Jetson Orin (CUDA arch 87); this takes a while the first time ..."
    CUDA_ARCH=87 scripts/build_llama_cpp.sh
  fi

  port_busy 8080 && die "Port 8080 is already in use. Stop whatever is using it and retry."
  say "Serving Qwen3.6-35B-A3B with experts paged from the SSD; Ctrl+C stops it."
  exec .venv/bin/python -m engine.serve --cpu-moe --ctx "${FAST_MOE_NX_CTX:-8192}" ${llama_args[@]+-- "${llama_args[@]}"}
}

if [[ $nx -eq 1 ]]; then
  run_nx
elif [[ ${#llama_args[@]} -gt 0 ]]; then
  die "'--' llama-server arguments only work with --nx."
fi

if [[ -n "${FAST_MOE_RAM_LIMIT:-}" ]]; then
  [[ "$FAST_MOE_RAM_LIMIT" =~ ^[1-9][0-9]*[gGmM]$ ]] || die "RAM limit must look like 20g or 16384m, got '$FAST_MOE_RAM_LIMIT'."
  export FAST_MOE_RAM_LIMIT
fi

command -v docker >/dev/null || die "Docker is not installed: https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required ('docker compose')."
docker info >/dev/null 2>&1 || die "Cannot reach the Docker daemon. Is Docker running, and is your user in the 'docker' group?"

gpu_ready() {
  command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

gpu_problem() {
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "no NVIDIA GPU is visible (nvidia-smi fails; the driver may be missing or the GPU switched off)"
  else
    echo "Docker has no NVIDIA runtime; install the NVIDIA Container Toolkit (see README)"
  fi
}

case "$mode" in
  auto)
    if gpu_ready; then
      mode="gpu"
    else
      warn "Using CPU mode: $(gpu_problem)."
      mode="cpu"
    fi
    ;;
  gpu) gpu_ready || die "GPU mode unavailable: $(gpu_problem). Try './start.sh cpu'." ;;
esac

if [[ "$mode" == "gpu" ]]; then
  compose_file="$ROOT/docker-compose.yml"
else
  compose_file="$ROOT/docker-compose.cpu.yml"
fi
compose=(docker compose -f "$compose_file")

for port in 7860 8080; do
  port_busy "$port" && die "Port $port is already in use. Stop whatever is using it (e.g. an earlier Fast-MoE) and retry."
done

# Pre-create the models folder so Docker doesn't create it owned by root.
mkdir -p "$ROOT/models"
export FAST_MOE_UID="${FAST_MOE_UID:-$(id -u)}"
export FAST_MOE_GID="${FAST_MOE_GID:-$(id -g)}"

stopped=0
cleanup() {
  [[ $stopped -eq 1 ]] && return
  stopped=1
  trap - INT TERM EXIT
  echo
  say "Stopping Fast-MoE (docker compose down)..."
  "${compose[@]}" down --timeout 30 || warn "docker compose down reported an error."
  say "Stopped."
}
trap cleanup EXIT
trap 'exit 130' INT TERM

up_args=(up --detach)
[[ $build -eq 1 ]] && up_args+=(--build)
say "Starting Fast-MoE in $mode mode${FAST_MOE_RAM_LIMIT:+ with a ${FAST_MOE_RAM_LIMIT} RAM limit}..."
"${compose[@]}" "${up_args[@]}"

say "Waiting for the dashboard at $UI_URL ..."
deadline=$((SECONDS + WAIT_SECONDS))
until curl -fsS -o /dev/null "$UI_URL/"; do
  if [[ -z "$("${compose[@]}" ps --status running --quiet)" ]]; then
    "${compose[@]}" logs --tail 40 || true
    die "The Fast-MoE container stopped during startup (logs above)."
  fi
  (( SECONDS < deadline )) || die "The dashboard did not come up within ${WAIT_SECONDS}s."
  sleep 2
done

say "Dashboard ready: $UI_URL"
say "OpenAI-compatible API (once a model is started): $API_URL"
if [[ $open_browser -eq 1 ]]; then
  if command -v xdg-open >/dev/null && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    xdg-open "$UI_URL" >/dev/null 2>&1 || warn "Could not open a browser; open $UI_URL yourself."
  elif command -v open >/dev/null; then
    open "$UI_URL" >/dev/null 2>&1 || warn "Could not open a browser; open $UI_URL yourself."
  else
    say "Open $UI_URL in your browser."
  fi
fi

say "Showing logs. Press Ctrl+C to stop everything."
"${compose[@]}" logs --follow --since 0s || true
