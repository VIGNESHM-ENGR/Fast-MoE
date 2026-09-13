# Multi-stage build for Fast-MoE. GPU-enabled by default; the `cpu` target
# provides a CUDA-free fallback. See TASKS.md milestone M7 — this is
# scaffolding only until M4 pins kt-kernel + sglang-kt and M2-M6 land
# real engine code to install.
#
# Build:  docker build --target gpu -t fast-moe:gpu .
#         docker build --target cpu -t fast-moe:cpu .

# ---- builder (GPU): compiles KTransformers' CUDA/AMX extensions + Fast-MoE
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY engine ./engine
COPY ui ./ui

# TODO(M7): install a pinned KTransformers commit/tag here (compiles its
# CUDA/AMX extensions), then `pip install .`

# ---- runtime (GPU)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-gpu /build /app
WORKDIR /app

EXPOSE 8000 7860
ENTRYPOINT ["python3", "-m", "engine.serve"]

# ---- runtime (CPU-only fallback)
FROM python:3.11-slim AS cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY engine ./engine
COPY ui ./ui

# TODO(M7): pip install .[cpu] once a CPU-only extra is defined

EXPOSE 8000 7860
ENTRYPOINT ["python3", "-m", "engine.serve"]
