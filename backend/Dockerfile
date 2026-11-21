# Red Box API image. Build from the repository root:
#   docker build -f backend/Dockerfile -t redbox-api .
# The model bundle is mounted at /app/bundle (see docker-compose.yml) or baked in for deployment.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    REDBOX_BUNDLE_DIR=/app/bundle

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

# LightGBM needs the OpenMP runtime, which the slim base image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached unless the lock files change). PyTorch is never installed: the API only
# needs the recsys package's core dependencies (NumPy, SciPy, LightGBM, ...).
WORKDIR /app
COPY ml/pyproject.toml ml/uv.lock ml/README.md ml/
COPY ml/src ml/src
COPY backend/pyproject.toml backend/uv.lock backend/
WORKDIR /app/backend
RUN uv sync --frozen --no-dev --no-cache

COPY backend/app app

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 api && chown -R api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
