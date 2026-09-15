#!/usr/bin/env bash
# Start Fast-MoE in Docker, open the dashboard, and stop everything on Ctrl+C.
#
#   ./start.sh              auto: GPU if Docker can use an NVIDIA GPU, otherwise CPU
#   ./start.sh gpu          NVIDIA GPU (docker-compose.yml)
#   ./start.sh cpu          CPU only (docker-compose.cpu.yml)
#   ./start.sh --no-browser --no-build --ram-limit 20g
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_URL="http://127.0.0.1:7860"
API_URL="http://127.0.0.1:8080/v1"
WAIT_SECONDS="${FAST_MOE_START_TIMEOUT:-600}"

mode="auto"
open_browser=1
build=1

usage() {
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  auto | gpu | cpu   device mode (default: auto)
  --no-browser       don't open the dashboard in a browser
  --no-build         start the existing image without rebuilding
  --ram-limit SIZE   cap the container's RAM, e.g. 20g or 16384m (default: no cap)
  -h, --help         show this help

Environment:
  FAST_MOE_START_TIMEOUT  seconds to wait for the dashboard (default 600)
  FAST_MOE_RAM_LIMIT      same as --ram-limit
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
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

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

port_busy() {
  if command -v ss >/dev/null; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}
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
