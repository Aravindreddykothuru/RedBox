# Red Box API for a Hugging Face Space (Docker SDK, free CPU tier: 2 vCPU, 16 GB RAM).
# This file is copied to the root of the Space by deploy/huggingface/build_space.py, next to:
#   ml/ (the recsys package), backend/ (the API) and bundle/ (the exported models, baked into the image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    REDBOX_BUNDLE_DIR=/app/bundle \
    # Space secrets/variables override these at runtime. Without a database URL the API uses a SQLite
    # file in /tmp, which is wiped whenever the Space restarts (fine for a demo, use Supabase otherwise).
    REDBOX_DATABASE_URL=sqlite:////tmp/redbox.db \
    # the poster cache (app/posters.py) refills itself as posters are viewed after a restart
    REDBOX_POSTER_DIR=/tmp/posters

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

# Spaces run containers as user 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user ml/pyproject.toml ml/uv.lock ml/README.md ml/
COPY --chown=user ml/src ml/src
COPY --chown=user backend/pyproject.toml backend/uv.lock backend/
WORKDIR /app/backend
# --no-cache: keep uv's download cache (~600 MB) out of the image
RUN uv sync --frozen --no-dev --no-cache && chown -R user /app
# LightGBM needs the OpenMP runtime, which the slim base image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --chown=user backend/app app
COPY --chown=user bundle /app/bundle

USER user
EXPOSE 7860
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
