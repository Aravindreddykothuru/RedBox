"""Glue between the database (what users did) and the recommendation engine (what to show next)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, HTTPException, Request, status
from recsys.data.tmdb import poster_url
from recsys.serving.engine import Engine
from recsys.serving.engine import Recommendation as EngineRec
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import schemas
from app.db import Database, Event, Profile, Rating, WatchedItem


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_db(request: Request):
    yield from request.app.state.db.session()


def update_new_films_later(request: Request, background: BackgroundTasks) -> None:
    """After the response: re-estimate the new films from the app's ratings of them (app/releases.py),
    because this request changed what someone thinks of one."""
    releases = getattr(request.app.state, "releases", None)
    if releases is not None:
        background.add_task(releases.update_feedback)


def item_index(engine: Engine, movie_id: int) -> int:
    idx = engine.movie_to_idx.get(movie_id)
    if idx is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown movie {movie_id}")
    return idx


def aware(t: datetime) -> datetime:
    """SQLite hands datetimes back without a time zone (they are stored as UTC); Postgres with one."""
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def user_history(
    db: Session, engine: Engine, user_id: uuid.UUID, threshold: float
) -> tuple[list[int], set[int]]:
    """(liked items in the order the user met them, every item they rated or watched). The order matters
    for the sequential models; the full set is excluded from recommendations.

    Liked means rated at or above the threshold, or watched (imported history: implicit feedback) and not
    rated below it. A film both watched and rated takes the earlier of the two times."""
    when: dict[int, datetime] = {}
    stars: dict[int, float] = {}
    for m, r, t in db.execute(
        select(Rating.movie_id, Rating.rating, Rating.created_at).where(Rating.user_id == user_id)
    ):
        stars[m], when[m] = r, aware(t)
    for m, t in db.execute(
        select(WatchedItem.movie_id, WatchedItem.watched_at).where(WatchedItem.user_id == user_id)
    ):
        when[m] = min(when[m], aware(t)) if m in when else aware(t)
    idx = engine.movie_to_idx
    order = sorted(when, key=lambda m: (when[m], m))
    # films the models know only: a new film (serving/newfilms.py) has no factors in any generator
    liked = [
        idx[m] for m in order if m in idx and idx[m] < engine.n_items and stars.get(m, threshold) >= threshold
    ]
    seen = {idx[m] for m in when if m in idx}
    return liked, seen


def favorite_genres(db: Session, user_id: uuid.UUID) -> list[str]:
    """The genres a user picked at onboarding: what a brand-new user's first picks are predicted from."""
    profile = db.get(Profile, user_id)
    return list(profile.favorite_genres) if profile and profile.favorite_genres else []


def _opt(row, column: str):
    """Optional catalog column (e.g. TMDB fields), None when absent or missing."""
    if column not in row.index:
        return None
    value = row[column]
    return (
        None
        if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value))
        else value
    )


def movie(engine: Engine, item_idx: int) -> schemas.Movie:
    row = engine.items.iloc[int(item_idx)]
    year = _opt(row, "year")
    return schemas.Movie(
        movie_id=int(row["movie_id"]),
        title=str(row["title"]),
        year=None if year is None else int(year),
        genres=list(row["genres"]),
        popularity=int(row["popularity"]),
        # Wikipedia posters go through the API's cache (app/posters.py): a path on the API, which the
        # website prefixes with the API's address. TMDB's image server needs no cache.
        poster_url=f"/posters/{int(row['movie_id'])}.webp"
        if _opt(row, "poster")
        else poster_url(_opt(row, "poster_path")),
    )


def movie_detail(engine: Engine, item_idx: int) -> schemas.MovieDetail:
    row = engine.items.iloc[int(item_idx)]
    cast = _opt(row, "cast")
    keywords = _opt(row, "keywords")
    imdb = _opt(row, "imdb_id")
    return schemas.MovieDetail(
        **movie(engine, item_idx).model_dump(),
        overview=_opt(row, "overview"),
        overview_source=_opt(row, "overview_source"),
        language=_opt(row, "language"),
        tagline=_opt(row, "tagline"),
        runtime=None if _opt(row, "runtime") is None else int(_opt(row, "runtime")),
        cast=[] if cast is None else [str(c) for c in cast],
        keywords=[] if keywords is None else [str(k) for k in keywords],
        avg_rating=None if _opt(row, "avg_rating") is None else round(float(_opt(row, "avg_rating")), 2),
        n_ratings=None if _opt(row, "n_ratings") is None else int(_opt(row, "n_ratings")),
        tmdb_id=None if _opt(row, "tmdb_id") is None else int(_opt(row, "tmdb_id")),
        imdb_id=None if imdb is None else f"tt{str(imdb).zfill(7)}",
    )


def rec(engine: Engine, r: EngineRec) -> schemas.Recommendation:
    return schemas.Recommendation(**movie(engine, r.item_idx).model_dump(), score=r.score, reason=r.reason)


def recs(engine: Engine, items: Iterable[EngineRec]) -> list[schemas.Recommendation]:
    return [rec(engine, r) for r in items]


def log_impressions(
    db: Database,
    user_id: uuid.UUID,
    rows: dict[str, list[schemas.Recommendation]],
    variant: str | None = None,
) -> None:
    """Logs what was shown, per source (a model or a feed row). Runs after the response is sent
    (FastAPI background task), so it never slows a request down."""
    with db.sessions() as s:
        s.add_all(
            Event(
                user_id=user_id,
                movie_id=r.movie_id,
                type="impression",
                source_model=source,
                position=i,
                variant=variant,
            )
            for source, shown in rows.items()
            for i, r in enumerate(shown)
        )
        s.commit()
