"""Import a watch history from Netflix, IMDb or Letterboxd: preview the matches, then save the films the
user keeps. The uploaded file is read in memory and never stored (see app/history_import.py)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from recsys.serving.engine import Engine
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import history_import, schemas
from app.auth import current_user
from app.db import Profile, Rating, WatchedItem, WatchlistItem, now
from app.services import aware, get_db, get_engine, item_index, movie, update_new_films_later

router = APIRouter(tags=["import"])

Kind = Literal["rating", "watched", "watchlist"]
Source = Literal["netflix", "imdb", "letterboxd"]


class ImportFileIn(BaseModel):
    name: str = Field(
        min_length=1, max_length=200, description="the file's name, e.g. NetflixViewingHistory.csv"
    )
    content: str = Field(
        max_length=14_500_000, description="the file: its text, or base64 for a .zip (10 MB at most)"
    )
    encoding: Literal["text", "base64"] = "text"


class ImportMatch(BaseModel):
    movie: schemas.Movie
    kind: Kind
    rating: float | None
    at: datetime | None
    source_title: str = Field(description="the title as the other service wrote it")
    times: int = Field(description="how often it was watched or rated (rewatches)")
    already: bool = Field(description="already rated or imported in Red Box")


class ImportPreviewOut(BaseModel):
    source: Source
    rows: int = Field(description="usable lines in the file")
    tv_skipped: int = Field(description="TV episodes and series: Red Box recommends films")
    short_skipped: int = Field(
        description="films watched for less than 15 minutes in all (full Netflix download)"
    )
    unmatched_count: int = Field(description="titles that are not in the catalogue")
    unmatched_sample: list[str]
    items: list[ImportMatch]


class ImportItemIn(BaseModel):
    movie_id: int
    kind: Kind
    rating: float | None = Field(default=None, ge=0.5, le=5.0)
    at: datetime | None = None
    times: int = Field(default=1, ge=1, le=10_000)

    @field_validator("rating")
    @classmethod
    def half_stars(cls, v: float | None) -> float | None:
        if v is not None and round(v * 2) != v * 2:
            raise ValueError("ratings go in half-star steps")
        return v


class ImportIn(BaseModel):
    source: Source
    items: list[ImportItemIn] = Field(max_length=5000)


class ImportResultOut(BaseModel):
    ratings: int
    watched: int
    watchlist: int
    unchanged: int = Field(description="already in Red Box: existing ratings are never overwritten")


class WatchedMovie(schemas.Movie):
    watched_at: datetime
    source: str
    times: int


@router.post("/me/import/preview", response_model=ImportPreviewOut)
def preview_import(
    body: ImportFileIn,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> ImportPreviewOut:
    """Read an exported watch history and match it to the catalogue. Nothing is saved yet."""
    try:
        p = history_import.preview(body.name, body.content, body.encoding, engine.items)
    except history_import.ImportFileError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    known = set(db.scalars(select(Rating.movie_id).where(Rating.user_id == user))) | set(
        db.scalars(select(WatchedItem.movie_id).where(WatchedItem.user_id == user))
    )
    return ImportPreviewOut(
        source=p.source,
        rows=p.rows,
        tv_skipped=p.tv_skipped,
        short_skipped=p.short_skipped,
        unmatched_count=len(p.unmatched),
        unmatched_sample=p.unmatched[:20],
        items=[
            ImportMatch(
                movie=movie(engine, engine.movie_to_idx[m.movie_id]),
                kind=m.kind,
                rating=m.rating,
                at=m.at,
                source_title=m.source_title,
                times=m.times,
                already=m.movie_id in known,
            )
            for m in p.matches
            if m.movie_id in engine.movie_to_idx
        ],
    )


@router.post("/me/import", response_model=ImportResultOut)
def run_import(
    body: ImportIn,
    request: Request,
    background: BackgroundTasks,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> ImportResultOut:
    """Save the films the user kept from a preview, with their original dates, so the sequential models
    see them in the order they were watched. Existing ratings are never overwritten; a film already on My
    list or already watched is not added to My list again. Counts as onboarding."""
    update_new_films_later(request, background)  # the history may include new films
    stamp = now()
    rated = set(db.scalars(select(Rating.movie_id).where(Rating.user_id == user)))
    listed = set(db.scalars(select(WatchlistItem.movie_id).where(WatchlistItem.user_id == user)))
    watched = {w.movie_id: w for w in db.scalars(select(WatchedItem).where(WatchedItem.user_id == user))}
    counts = {"ratings": 0, "watched": 0, "watchlist": 0, "unchanged": 0}
    for it in {it.movie_id: it for it in reversed(body.items)}.values():  # one entry per film
        item_index(engine, it.movie_id)
        at = min(aware(it.at), stamp) if it.at else stamp
        if it.kind == "rating":
            if it.rating is None:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, f"film {it.movie_id}: a rating needs stars"
                )
            if it.movie_id in rated:
                counts["unchanged"] += 1
                continue
            db.add(
                Rating(user_id=user, movie_id=it.movie_id, rating=it.rating, created_at=at, updated_at=stamp)
            )
            rated.add(it.movie_id)
            counts["ratings"] += 1
        elif it.kind == "watched":
            row = watched.get(it.movie_id)
            if row is not None:
                row.watched_at = max(aware(row.watched_at), at)
                row.times = max(row.times, it.times)
                counts["unchanged"] += 1
                continue
            watched[it.movie_id] = WatchedItem(
                user_id=user, movie_id=it.movie_id, watched_at=at, source=body.source, times=it.times
            )
            db.add(watched[it.movie_id])
            counts["watched"] += 1
        else:
            if it.movie_id in listed or it.movie_id in rated or it.movie_id in watched:
                counts["unchanged"] += 1
                continue
            db.add(WatchlistItem(user_id=user, movie_id=it.movie_id, added_at=at))
            listed.add(it.movie_id)
            counts["watchlist"] += 1
    profile = db.get(Profile, user) or Profile(user_id=user)
    if profile.onboarded_at is None:
        profile.onboarded_at = stamp
    db.add(profile)
    db.commit()
    return ImportResultOut(**counts)


@router.get("/me/watched", response_model=list[WatchedMovie])
def my_watched(
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> list[WatchedMovie]:
    """The imported watch history, latest first."""
    rows = db.scalars(
        select(WatchedItem).where(WatchedItem.user_id == user).order_by(WatchedItem.watched_at.desc())
    ).all()
    return [
        WatchedMovie(
            **movie(engine, engine.movie_to_idx[w.movie_id]).model_dump(),
            watched_at=w.watched_at,
            source=w.source,
            times=w.times,
        )
        for w in rows
        if w.movie_id in engine.movie_to_idx
    ]


@router.delete("/me/watched/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def forget_watched(
    movie_id: int,
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> Response:
    db.execute(delete(WatchedItem).where(WatchedItem.user_id == user, WatchedItem.movie_id == movie_id))
    db.commit()
    update_new_films_later(request, background)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/me/watched", status_code=status.HTTP_204_NO_CONTENT)
def clear_watched(
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> Response:
    """Forget the whole imported watch history (ratings and My list stay)."""
    db.execute(delete(WatchedItem).where(WatchedItem.user_id == user))
    db.commit()
    update_new_films_later(request, background)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
