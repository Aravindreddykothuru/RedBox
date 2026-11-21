# Red Box API on Render (render.yaml at the repository root). Built from the repository itself; the model
# bundle is not in git (MovieLens data may not be redistributed): start.py downloads it from a private
# Hugging Face dataset onto the service's persistent disk the first time, and reuses it afterwards.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
# LightGBM needs the OpenMP runtime, which the slim base image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ml/pyproject.toml ml/uv.lock ml/README.md ml/
COPY ml/src ml/src
COPY backend/pyproject.toml backend/uv.lock backend/
WORKDIR /app/backend
# --no-cache: keep uv's download cache out of the image
RUN uv sync --frozen --no-dev --no-cache

COPY backend/app app
COPY deploy/render/start.py /app/start.py

CMD ["uv", "run", "--no-sync", "python", "/app/start.py"]
