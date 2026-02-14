# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build tools for packages with C/Cython extensions (hdbscan, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

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

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system deps:
#   libpq5   - asyncpg PostgreSQL driver
#   libgomp1 - OpenMP for torch/numpy
#   curl     - healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv

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
