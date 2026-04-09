# Build context: repo root (docker compose sets context: .)
# The image is NOT platform-specific for the Python deps, but the pre-built
# _ccr_engine.so IS platform-specific (arm64 macOS).  For production on Linux,
# rebuild the engine bindings first: ./scripts/build_engine.sh --bindings

FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq-dev \
       gcc \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Python deps (cached layer — only rebuilt when lock file changes) ──────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Application source ────────────────────────────────────────────────────────
COPY server/ ./server/

# ── Pre-built SvelteKit frontend (optional — served at / when present) ────────
# web/build is .gitignored.  Build with: cd web && npm run build
# The build output is then copied by docker-compose volume mount in dev,
# or by a COPY instruction in a production multi-stage build.

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
