"""Red Box API: run with `uv run uvicorn app.main:app --reload` (docs at /docs)."""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from recsys.serving.engine import Engine

from app import __version__
from app.api import accounts, catalog, group, imports, me, ops, posters, recs
from app.backup import Backup
from app.db import Database
from app.releases import Releases
from app.security import RateLimiter
from app.settings import Settings, get_settings

log = logging.getLogger("redbox")
access_log = logging.getLogger("redbox.access")  # paths only: no query strings, bodies or tokens


def _show_app_logs() -> None:
    """Print the API's own log lines (requests, new releases, backups) next to uvicorn's: by default only
    warnings would reach the console, and a hosted Space's log shows only the console."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _show_app_logs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model bundle once at startup; every request reuses it (pure NumPy, thread-safe reads).
        app.state.engine = Engine(settings.bundle_dir)
        app.state.backup = Backup(settings)
        app.state.backup.restore()  # a host without a permanent disk: bring back the last copy first
        app.state.db = Database(settings.database_url)
        app.state.db.create_tables()
        app.state.jwt_secret = settings.jwt_secret or secrets.token_urlsafe(48)
        if settings.auth_mode == "local" and not settings.jwt_secret:
            log.warning("REDBOX_JWT_SECRET is not set: sign-ins will not survive a restart")
        app.state.login_limiter = RateLimiter(settings.login_max_failures, settings.login_window_minutes * 60)
        log.info(
            "bundle %s loaded: %d items, models %s",
            settings.bundle_dir,
            app.state.engine.n_items,
            app.state.engine.models,
        )
        # new releases: warm up the new films from the ratings so far, then look for more once a day
        app.state.releases = Releases(app.state.engine, app.state.db, settings)
        app.state.releases.update_feedback()
        app.state.releases.start()
        app.state.backup.start()
        yield
        app.state.releases.stop()
        app.state.backup.stop()

    app = FastAPI(
        title="Red Box API",
        version=__version__,
        description="Personalized movie recommendations from a two-stage hybrid recommender.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_and_harden(request: Request, call_next):
        """One log line per request (method, path, status, duration) and standard security headers."""
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Server-Timing"] = f"app;dur={ms:.1f}"
        access_log.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, ms)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (
        catalog.router,
        accounts.router,
        me.router,
        imports.router,
        recs.router,
        group.router,
        ops.router,
        posters.router,
    ):
        app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


app = create_app()
