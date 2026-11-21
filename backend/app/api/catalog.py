"""Service status and the movie catalog (browse, search, details, similar movies)."""

from __future__ import annotations

import uuid
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from recsys.serving.engine import Engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import schemas
from app.auth import optional_user
from app.db import Rating, WatchlistItem
from app.services import get_db, get_engine, item_index, movie, movie_detail, recs

router = APIRouter()


@router.get("/health", response_model=schemas.Health, tags=["status"])
def health(engine: Engine = Depends(get_engine)) -> schemas.Health:
    m = engine.manifest
    return schemas.Health(
        status="ok",
        dataset=m.get("dataset", "?"),
        bundle_created=m.get("created", "?"),
        n_items=engine.n_items,
        models=engine.models,
        semantic_search=engine.has_text,
        serving_test_metrics={k: v for k, v in m.get("test_metrics", {}).items() if k.endswith("@10")},
    )


@router.get("/movies/genres", response_model=list[str], tags=["movies"])
def genres(engine: Engine = Depends(get_engine)) -> list[str]:
    return sorted({g for gs in engine.items["genres"] for g in gs})


@router.get("/movies/trending", response_model=list[schemas.Recommendation], tags=["movies"])
def trending(
    k: int = Query(10, ge=1, le=50), engine: Engine = Depends(get_engine)
) -> list[schemas.Recommendation]:
    """What the hybrid predicts many viewers will watch next (the crowd forecast), for anyone."""
    return recs(engine, engine.trending(k=k))


@router.get("/movies", response_model=schemas.MoviePage, tags=["movies"])
def list_movies(
    q: str | None = Query(None, max_length=100, description="Words that must appear in the title"),
    genre: str | None = None,
    sort: Literal["popular", "newest", "title"] = "popular",
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(get_engine),
) -> schemas.MoviePage:
    items = engine.items
    mask = np.ones(len(items), dtype=bool)
    if q:
        titles = items["title"].str.lower()
        for word in q.lower().split():
            mask &= titles.str.contains(word, regex=False).to_numpy()
    if genre:
        mask &= items["genres"].map(lambda gs: genre in gs).to_numpy()
    found = items[mask]
    if sort == "popular":
        found = found.sort_values("popularity", ascending=False, kind="stable")
    elif sort == "newest":
        found = found.sort_values(["year", "popularity"], ascending=False, kind="stable", na_position="last")
    else:
        found = found.sort_values("title", kind="stable")
    page = found.iloc[offset : offset + limit]
    return schemas.MoviePage(total=len(found), items=[movie(engine, i) for i in page["item_idx"]])


@router.get("/movies/{movie_id}", response_model=schemas.MovieDetail, tags=["movies"])
def get_movie(
    movie_id: int,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID | None = Depends(optional_user),
) -> schemas.MovieDetail:
    detail = movie_detail(engine, item_index(engine, movie_id))
    if user is not None:
        detail.your_rating = db.scalar(
            select(Rating.rating).where(Rating.user_id == user, Rating.movie_id == movie_id)
        )
        detail.in_watchlist = (
            db.scalar(
                select(WatchlistItem.movie_id).where(
                    WatchlistItem.user_id == user, WatchlistItem.movie_id == movie_id
                )
            )
            is not None
        )
    return detail


@router.get("/movies/{movie_id}/similar", response_model=list[schemas.Recommendation], tags=["movies"])
def similar(
    movie_id: int,
    mode: Literal["cf", "content"] = "cf",
    k: int = Query(12, ge=1, le=50),
    engine: Engine = Depends(get_engine),
) -> list[schemas.Recommendation]:
    """'cf': liked by the same viewers (collaborative filtering); 'content': similar story and themes
    (text embeddings), or shared genres and era when the bundle has no text model."""
    return recs(engine, engine.similar(item_index(engine, movie_id), k=k, mode=mode))


@router.get("/search/semantic", response_model=list[schemas.Recommendation], tags=["movies"])
def semantic_search(
    q: str = Query(..., min_length=2, max_length=200, description='e.g. "mind-bending space movies"'),
    k: int = Query(24, ge=1, le=100),
    engine: Engine = Depends(get_engine),
) -> list[schemas.Recommendation]:
    """Natural-language search: the query and every movie's text (title, genres, viewer keywords, plot)
    are embedded with the same sentence encoder and compared by cosine similarity."""
    if not engine.has_text:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "semantic search is not enabled for this model bundle"
        )
    return recs(engine, engine.semantic_search(q, k=k))
