# Fast-MoE on top of the official llama.cpp server image.
#
# The control panel starts and restarts llama-server itself (with the placement
# llama-fit-params computes), so both live in one image. BASE picks the backend:
#   CUDA (default): ghcr.io/ggml-org/llama.cpp:server-cuda-b10920
#   CPU:            ghcr.io/ggml-org/llama.cpp:server-b10920
ARG BASE=ghcr.io/ggml-org/llama.cpp:server-cuda-b10920
FROM ${BASE}

# The base image has Python 3.12 but no venv/ensurepip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/fast-moe
ENV PATH=/opt/fast-moe/bin:$PATH

WORKDIR /opt/fast-moe/src
COPY pyproject.toml README.md LICENSE ./
COPY engine ./engine
COPY ui ./ui
RUN pip install --no-cache-dir .

# fit-params ships as a subcommand of the image's `llama` tool; expose it under
# the standalone name Fast-MoE looks up.
RUN printf '#!/bin/sh\nexec /app/llama fit-params "$@"\n' > /usr/local/bin/llama-fit-params \
    && chmod 755 /usr/local/bin/llama-fit-params

# The llama.cpp binaries have no RPATH and only find their .so files next to them
# via the library path, so make that work from any working directory.
ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}

# Servers listen on all interfaces inside the container; compose publishes them on
# 127.0.0.1 only. HOME=/tmp keeps logs writable when running as a non-root user.
ENV LLAMA_CPP_BIN_DIR=/app \
    FAST_MOE_MODELS_DIR=/models \
    FAST_MOE_UI_HOST=0.0.0.0 \
    FAST_MOE_API_HOST=0.0.0.0 \
    HOME=/tmp \
    PYTHONUNBUFFERED=1

WORKDIR /opt/fast-moe
EXPOSE 7860 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:7860/ >/dev/null || exit 1

ENTRYPOINT []
CMD ["python", "-m", "ui.app"]
