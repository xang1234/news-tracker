# syntax=docker/dockerfile:1.6
# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build tools for packages with C/Cython extensions (hdbscan, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

ARG XUI_INSTALL=true
ARG XUI_PIP_SPEC=git+https://github.com/xang1234/xui.git@main
ARG EXPORT_ONNX_MODELS=true
ARG CPU_RUNTIME_OPTIMIZED=false

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies only (frozen from lockfile)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself
COPY src/ src/
RUN uv sync --frozen --no-dev
RUN mkdir -p /app/models

# Install ONNX tooling in the builder and export CPU-optimized model artifacts.
RUN if [ "$EXPORT_ONNX_MODELS" = "true" ]; then \
      set -eux; \
      uv venv /tmp/onnx-export; \
      uv pip install -p /tmp/onnx-export/bin/python \
        "torch>=2.4.0" \
        "transformers>=4.47.1" \
        "optimum-onnx[onnxruntime]>=0.0.3"; \
      mkdir -p /app/models/embedding-finbert /app/models/embedding-minilm /app/models/sentiment-finbert; \
      /tmp/onnx-export/bin/optimum-cli export onnx --model ProsusAI/finbert --task feature-extraction /app/models/embedding-finbert; \
      /tmp/onnx-export/bin/optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction /app/models/embedding-minilm; \
      /tmp/onnx-export/bin/optimum-cli export onnx --model ProsusAI/finbert --task text-classification /app/models/sentiment-finbert; \
      rm -rf /tmp/onnx-export; \
    fi

# Install xui CLI from git (private/public) when enabled.
# Private token is passed as a BuildKit secret (id=xui_github_token) to avoid leaking in logs.
RUN --mount=type=secret,id=xui_github_token \
    if [ "$XUI_INSTALL" = "true" ]; then \
      spec="$XUI_PIP_SPEC"; \
      token_file="/run/secrets/xui_github_token"; \
      token=""; \
      if [ -f "$token_file" ]; then \
        token="$(tr -d '\r\n' < "$token_file")"; \
      fi; \
      if [ -n "$token" ] && echo "$spec" | grep -q '^git+https://github.com/'; then \
        spec="$(echo "$spec" | sed "s#^git+https://github.com/#git+https://${token}@github.com/#")"; \
      fi; \
      uv pip install -p /app/.venv/bin/python -n "xui-reader[cli] @ ${spec}"; \
    fi

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system deps:
#   libpq5/libgomp1/curl - app runtime and healthcheck
#   Playwright Chromium deps for xui browser automation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libpq5 \
      libgomp1 \
      curl \
      ca-certificates \
      fonts-liberation \
      libasound2 \
      libatk-bridge2.0-0 \
      libatk1.0-0 \
      libcairo2 \
      libcups2 \
      libdbus-1-3 \
      libdrm2 \
      libgbm1 \
      libglib2.0-0 \
      libgtk-3-0 \
      libnspr4 \
      libnss3 \
      libpango-1.0-0 \
      libx11-6 \
      libx11-xcb1 \
      libxcb1 \
      libxcomposite1 \
      libxdamage1 \
      libxext6 \
      libxfixes3 \
      libxkbcommon0 \
      libxrandr2 && \
    rm -rf /var/lib/apt/lists/*

ARG XUI_INSTALL=true
ARG CPU_RUNTIME_OPTIMIZED=false
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV EMBEDDING_ONNX_MODEL_PATH=/app/models/embedding-finbert
ENV EMBEDDING_ONNX_MINILM_MODEL_PATH=/app/models/embedding-minilm
ENV SENTIMENT_ONNX_MODEL_PATH=/app/models/sentiment-finbert

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models

# Prune Torch from dedicated CPU-only images only. The shared runtime image
# still supports optional Torch-backed features like fastcoref and spaCy trf.
RUN if [ "$CPU_RUNTIME_OPTIMIZED" = "true" ]; then \
      uv pip uninstall -p /app/.venv/bin/python torch optimum optimum-onnx onnx; \
    fi

# Install Playwright Chromium browser for xui runs.
RUN if [ "$XUI_INSTALL" = "true" ]; then \
      /app/.venv/bin/python -m playwright install chromium; \
    fi

# Copy application code and config
COPY src/ src/
COPY pyproject.toml ./
COPY migrations/ migrations/

# Put venv on PATH so `news-tracker` CLI is available
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root
USER appuser

EXPOSE 8001

# Health check with 60s start period for ML model lazy loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

# Flexible entrypoint: `docker run <image> sentiment-worker` overrides CMD
ENTRYPOINT ["news-tracker"]
CMD ["serve"]
