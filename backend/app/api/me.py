"""Everything a signed-in user owns: ratings, watchlist, profile and onboarding."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response, status
from recsys.data.wikipedia import NEW_ID_OFFSET
from recsys.serving.engine import Engine
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app import schemas
from app.auth import current_user
from app.db import Event, Profile, Rating, WatchedItem, WatchlistItem, now
from app.services import get_db, get_engine, item_index, movie, recs, update_new_films_later, user_history
from app.settings import Settings, get_settings

router = APIRouter(tags=["me"])


def _upsert_rating(db: Session, user: uuid.UUID, movie_id: int, value: float) -> Rating:
    row = db.get(Rating, (user, movie_id))
    if row is None:  # first rating: its time defines the position in the user's sequence
        row = Rating(user_id=user, movie_id=movie_id, rating=value)
        db.add(row)
    else:
        row.rating = value
    db.add(Event(user_id=user, movie_id=movie_id, type="rate"))
    return row


@router.put("/ratings/{movie_id}", response_model=schemas.RatedMovie)
def rate(
    movie_id: int,
    body: schemas.RatingIn,
    request: Request,
    background: BackgroundTasks,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> schemas.RatedMovie:
    """Rate a movie. The very next recommendation request already reflects it; a new film's rating also
    moves that film towards (or away from) viewers like you."""
    idx = item_index(engine, movie_id)
    row = _upsert_rating(db, user, movie_id, body.rating)
    db.commit()
    if movie_id >= NEW_ID_OFFSET:
        update_new_films_later(request, background)
    return schemas.RatedMovie(**movie(engine, idx).model_dump(), rating=row.rating, rated_at=row.created_at)


@router.delete("/ratings/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def unrate(
    movie_id: int,
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> Response:
    db.execute(delete(Rating).where(Rating.user_id == user, Rating.movie_id == movie_id))
    db.commit()
    if movie_id >= NEW_ID_OFFSET:
        update_new_films_later(request, background)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/ratings", response_model=list[schemas.RatedMovie])
def my_ratings(
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> list[schemas.RatedMovie]:
    rows = db.scalars(select(Rating).where(Rating.user_id == user).order_by(Rating.created_at.desc())).all()
    return [
        schemas.RatedMovie(
            **movie(engine, engine.movie_to_idx[r.movie_id]).model_dump(),
            rating=r.rating,
            rated_at=r.created_at,
        )
        for r in rows
        if r.movie_id in engine.movie_to_idx
    ]


@router.put("/watchlist/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_to_watchlist(
    movie_id: int,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> Response:
    item_index(engine, movie_id)
    if db.get(WatchlistItem, (user, movie_id)) is None:
        db.add(WatchlistItem(user_id=user, movie_id=movie_id))
        db.add(Event(user_id=user, movie_id=movie_id, type="watchlist"))
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/watchlist/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(
    movie_id: int, db: Session = Depends(get_db), user: uuid.UUID = Depends(current_user)
) -> Response:
    db.execute(delete(WatchlistItem).where(WatchlistItem.user_id == user, WatchlistItem.movie_id == movie_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/watchlist", response_model=list[schemas.Movie])
def my_watchlist(
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
) -> list[schemas.Movie]:
    ids = db.scalars(
        select(WatchlistItem.movie_id)
        .where(WatchlistItem.user_id == user)
        .order_by(WatchlistItem.added_at.desc())
    )
    return [movie(engine, engine.movie_to_idx[m]) for m in ids if m in engine.movie_to_idx]


@router.get("/me/profile", response_model=schemas.ProfileOut)
def my_profile(db: Session = Depends(get_db), user: uuid.UUID = Depends(current_user)) -> schemas.ProfileOut:
    p = db.get(Profile, user)
    n = db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user)) or 0
    watched = db.scalar(select(func.count()).select_from(WatchedItem).where(WatchedItem.user_id == user)) or 0
    return schemas.ProfileOut(
        display_name=p.display_name if p else None,
        favorite_genres=list(p.favorite_genres) if p else [],
        onboarded=bool(p and p.onboarded_at),
        n_ratings=n,
        n_watched=watched,
    )


@router.get("/onboarding/seeds", response_model=list[schemas.Movie], tags=["onboarding"])
def onboarding_seeds(
    n: int = Query(24, ge=1, le=60),
    genre: list[str] = Query(default=[], description="Prefer movies from these genres"),
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> list[schemas.Movie]:
    """Well-known but divisive movies: the ratings that tell us most about a new user's taste."""
    _, rated = user_history(db, engine, user, settings.positive_threshold)
    seeds = engine.onboarding_seeds(n=60, exclude=rated)
    if genre:  # chosen genres first, the rest after
        seeds.sort(key=lambda r: not set(genre) & set(r.genres))
    return [movie(engine, r.item_idx) for r in seeds[:n]]


@router.post("/onboarding", response_model=list[schemas.Recommendation], tags=["onboarding"])
def complete_onboarding(
    body: schemas.OnboardingIn,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> list[schemas.Recommendation]:
    """Save chosen genres and first ratings, and return the user's first recommendations."""
    profile = db.get(Profile, user) or Profile(user_id=user)
    profile.favorite_genres = body.favorite_genres
    profile.display_name = body.display_name or profile.display_name
    profile.onboarded_at = now()
    db.add(profile)
    for r in body.ratings:
        item_index(engine, r.movie_id)
        _upsert_rating(db, user, r.movie_id, r.rating)
    db.commit()
    liked, rated = user_history(db, engine, user, settings.positive_threshold)
    return recs(engine, engine.recommend(liked, rated, k=20, genres=body.favorite_genres))
