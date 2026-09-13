#!/usr/bin/env bash
# Build a pinned llama.cpp release natively (for running without Docker).
# Needs git, cmake >= 3.24, a C++ compiler, and the CUDA toolkit for GPU builds.
#
#   scripts/build_llama_cpp.sh              # CUDA build for this machine's GPU
#   GGML_CUDA=OFF scripts/build_llama_cpp.sh  # CPU-only build
set -euo pipefail

LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b10937}"
GGML_CUDA="${GGML_CUDA:-ON}"
CUDA_ARCH="${CUDA_ARCH:-native}"   # e.g. 86 for RTX 30xx, 89 for RTX 40xx
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"

if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin tag "$LLAMA_CPP_TAG"
  git -C "$SRC" checkout -q "$LLAMA_CPP_TAG"
fi

cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="$GGML_CUDA" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF
cmake --build "$SRC/build" --config Release -j "$(nproc)" --target llama-server llama-bench

echo "Built: $SRC/build/bin/llama-server"
