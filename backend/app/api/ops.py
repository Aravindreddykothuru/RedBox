"""Live metrics for operators: how real users respond to the recommendations (online evaluation)."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import uuid
from datetime import UTC, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import PlainTextResponse
from recsys.serving.engine import Engine
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app import experiments, schemas
from app.auth import admin_user
from app.db import Account, Event, Rating, now
from app.services import get_db, get_engine
from app.settings import Settings, get_settings

router = APIRouter(tags=["operations"])


def _rates(rows) -> dict[str, dict]:
    """{key: {impressions, clicks, ctr}} from rows of (key, type, count)."""
    out: dict[str, dict] = {}
    for key, kind, n in rows:
        entry = out.setdefault(key or "unknown", {"impressions": 0, "clicks": 0})
        if kind == "impression":
            entry["impressions"] += n
        elif kind == "click":
            entry["clicks"] += n
    for entry in out.values():
        entry["ctr"] = entry["clicks"] / entry["impressions"] if entry["impressions"] else None
    return out


@router.get("/ops/ratings.csv", response_class=PlainTextResponse)
def export_ratings(
    _: uuid.UUID = Depends(admin_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    """The app's ratings for retraining (`recsys import-app-ratings`): pseudonymous user keys (a keyed
    hash of the user id, so no email or id leaves the API), MovieLens movie ids, ratings, unix times."""
    key = (settings.jwt_secret or "redbox").encode()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["user_key", "movie_id", "rating", "timestamp"])
    for r in db.scalars(select(Rating).order_by(Rating.user_id, Rating.created_at)):
        user_key = hmac.new(key, str(r.user_id).encode(), hashlib.sha256).hexdigest()[:16]
        created = r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=UTC)
        writer.writerow([user_key, r.movie_id, r.rating, int(created.timestamp())])
    return PlainTextResponse(out.getvalue(), media_type="text/csv")


@router.get("/ops/metrics")
def live_metrics(
    days: int = Query(30, ge=1, le=365),
    _: uuid.UUID = Depends(admin_user),
    db: Session = Depends(get_db),
    engine: Engine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Click-through rate per feed row and per A/B-test group, activity and catalogue reach."""
    since = now() - timedelta(days=days)
    week = now() - timedelta(days=7)
    in_window = Event.created_at >= since
    per_source = db.execute(
        select(Event.source_model, Event.type, func.count())
        .where(in_window, Event.type.in_(["impression", "click"]))
        .group_by(Event.source_model, Event.type)
    ).all()
    # The experiment runs on the "Top picks" row (source "hybrid"); "manual" = users who set the slider.
    per_group = db.execute(
        select(Event.variant, Event.type, func.count())
        .where(in_window, Event.source_model == "hybrid", Event.type.in_(["impression", "click"]))
        .group_by(Event.variant, Event.type)
    ).all()
    groups = _rates(per_group)
    names = list(settings.ab_variants)
    test = None
    if len(names) >= 2:
        a, b = (groups.get(n, {"impressions": 0, "clicks": 0}) for n in names[:2])
        test = {
            "control": names[0],
            "treatment": names[1],
            **experiments.ctr_test(a["clicks"], a["impressions"], b["clicks"], b["impressions"]),
        }
    shown_items = db.scalar(
        select(func.count(distinct(Event.movie_id))).where(in_window, Event.type == "impression")
    )
    return {
        "window_days": days,
        "experiment": {
            "name": settings.ab_experiment,
            "groups": settings.ab_variants,
            "by_group": groups,
            "test": test,
        },
        "by_source": _rates(per_source),
        "activity": {
            "accounts": db.scalar(select(func.count()).select_from(Account)) or 0,
            "active_users_7d": db.scalar(
                select(func.count(distinct(Event.user_id))).where(Event.created_at >= week)
            )
            or 0,
            "ratings_7d": db.scalar(select(func.count()).select_from(Rating).where(Rating.updated_at >= week))
            or 0,
        },
        "catalogue": {
            "items": engine.n_items,
            "shown": shown_items or 0,
            "share_shown": (shown_items or 0) / engine.n_items,
        },
    }


@router.get("/ops/releases", response_model=schemas.ReleasesStatus)
def releases_status(request: Request, _: uuid.UUID = Depends(admin_user)) -> schemas.ReleasesStatus:
    """The new releases: how many, which languages, when the API last looked and what it found."""
    return schemas.ReleasesStatus(**request.app.state.releases.status())


@router.post("/ops/releases/refresh", response_model=schemas.ReleasesStatus, status_code=202)
def refresh_releases(
    request: Request, background: BackgroundTasks, _: uuid.UUID = Depends(admin_user)
) -> schemas.ReleasesStatus:
    """Look for new releases now instead of waiting for the daily look (it runs after this answer)."""
    background.add_task(request.app.state.releases.refresh)
    return schemas.ReleasesStatus(**request.app.state.releases.status())
