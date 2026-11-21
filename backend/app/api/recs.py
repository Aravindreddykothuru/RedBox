"""Recommendations, the home feed, the Model Lab and interaction events."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from recsys.serving.engine import Engine
from sqlalchemy.orm import Session

from app import experiments, schemas
from app.auth import current_user
from app.db import Event
from app.services import (
    favorite_genres,
    get_db,
    get_engine,
    item_index,
    log_impressions,
    movie,
    recs,
    user_history,
)
from app.settings import Settings, get_settings

router = APIRouter()


@router.get("/recommendations", response_model=list[schemas.Recommendation], tags=["recommendations"])
def recommendations(
    request: Request,
    background: BackgroundTasks,
    k: int = Query(20, ge=1, le=100),
    model: str = Query("hybrid", description="hybrid, or one generator for comparison (see /health)"),
    diversity: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="0 = most accurate; higher = more varied genres; omit to use the user's A/B-test group",
    ),
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> list[schemas.Recommendation]:
    if model not in engine.models:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"model must be one of {engine.models}")
    group = experiments.assign(user, settings)
    if diversity is None:  # the experiment decides; a user's own slider choice is logged as "manual"
        diversity = settings.ab_variants.get(group, 0.0) if model == "hybrid" else 0.0
    else:
        group = "manual"
    liked, rated = user_history(db, engine, user, settings.positive_threshold)
    genres = [] if liked else favorite_genres(db, user)  # only a brand-new user needs them
    out = recs(engine, engine.recommend(liked, rated, k=k, lam=1.0 - diversity, model=model, genres=genres))
    if settings.log_impressions:
        background.add_task(log_impressions, request.app.state.db, user, {model: out}, group)
    return out


@router.get("/recommendations/feed", response_model=schemas.Feed, tags=["recommendations"])
def feed(
    request: Request,
    background: BackgroundTasks,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> schemas.Feed:
    """The home page: personal rows when the user has liked something, then what many viewers are
    predicted to watch next (the crowd forecast)."""
    group = experiments.assign(user, settings)
    liked, rated = user_history(db, engine, user, settings.positive_threshold)
    rows: list[schemas.FeedRow] = []
    if liked:
        lam = 1.0 - settings.ab_variants.get(group, 0.0)
        rows.append(
            schemas.FeedRow(
                key="top_picks",
                title="Top picks for you",
                items=recs(engine, engine.recommend(liked, rated, k=20, lam=lam)),
            )
        )
        anchor, similar = engine.because_you_liked(liked, k=12, exclude=rated)
        if anchor is not None and similar:
            rows.append(
                schemas.FeedRow(
                    key="because_you_liked",
                    title=f"Because you liked {anchor.title}",
                    items=recs(engine, similar),
                )
            )
        gems = engine.hidden_gems(liked, k=12, exclude=rated)
        if gems:
            rows.append(
                schemas.FeedRow(key="hidden_gems", title="Hidden gems for you", items=recs(engine, gems))
            )
    rows.append(
        schemas.FeedRow(
            key="trending",
            title="Predicted to watch next",
            items=recs(engine, engine.trending(k=20, exclude=rated)),
        )
    )
    if settings.log_impressions:  # top picks are logged by /recommendations, which the app calls for them
        shown = {row.key: row.items for row in rows if row.key != "top_picks"}
        background.add_task(log_impressions, request.app.state.db, user, shown, group)
    return schemas.Feed(rows=rows)


@router.get("/recommendations/new-films", response_model=schemas.NewFilmsOut, tags=["recommendations"])
def new_films(
    request: Request,
    k: int = Query(20, ge=1, le=50),
    language: str | None = Query(None, description="one language's films only, e.g. Tamil"),
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> schemas.NewFilmsOut:
    """New releases: films newer than the models' data (English and Hindi since 2023; Telugu, Tamil,
    Kannada and Malayalam since 2016), found by the API's daily look at Wikidata. Ranked for you by their
    iALS factors: their story's guess, warmed by the app's ratings (app/releases.py). With no ratings yet,
    the most widely known releases, in the genres you picked first. All languages, or one with
    ``language``."""
    languages = engine.new_languages
    if language is not None and language not in languages:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"no new films in {language!r}; available: {', '.join(languages) or 'none'}",
        )
    liked, seen = user_history(db, engine, user, settings.positive_threshold)
    genres = None if liked else favorite_genres(db, user)
    items = engine.new_films(liked, k=k, exclude=seen, language=language, genres=genres)
    releases = getattr(request.app.state, "releases", None)
    return schemas.NewFilmsOut(
        languages=languages,
        language=language,
        refreshed_at=releases.refreshed_at if releases else None,
        items=recs(engine, items),
    )


@router.get("/recommendations/why/{movie_id}", response_model=schemas.WhyOut, tags=["recommendations"])
def why(
    movie_id: int,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> schemas.WhyOut:
    """Why the hybrid ranks this film where it does for you: the exact contribution of each signal to the
    LightGBM ranker's score (TreeSHAP), grouped by source, with the film's rank among your candidates and
    the models that proposed it."""
    idx = item_index(engine, movie_id)
    liked, _ = user_history(db, engine, user, settings.positive_threshold)
    found = engine.why(idx, liked)
    if found is None:
        return schemas.WhyOut(available=False)
    return schemas.WhyOut(
        available=True,
        rank=found["rank"],
        candidates=found["candidates"],
        proposed_by=found["proposed_by"],
        parts=[schemas.WhyPart(**p) for p in found["parts"]],
    )


@router.get("/lab/users", response_model=list[schemas.LabUser], tags=["model lab"])
def lab_users(
    limit: int = Query(50, ge=1, le=500), engine: Engine = Depends(get_engine)
) -> list[schemas.LabUser]:
    """Users of the training dataset whose recommendations can be inspected in the Model Lab."""
    ids = engine.dataset_user_ids()[:limit]
    return [schemas.LabUser(user_id=u, n_liked=len(engine.dataset_history(u))) for u in ids]


@router.get("/lab/compare", response_model=schemas.LabCompare, tags=["model lab"])
def lab_compare(
    k: int = Query(10, ge=1, le=50),
    dataset_user: int | None = Query(None, description="A training-dataset user; default = you"),
    diversity: float = Query(0.0, ge=0.0, le=1.0, description="MMR diversity for the hybrid column"),
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> schemas.LabCompare:
    """Every model's top-k for the same history, side by side."""
    if dataset_user is not None:
        try:
            liked = engine.dataset_history(dataset_user)
        except KeyError as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
        rated: set[int] = set(liked)
    else:
        liked, rated = user_history(db, engine, user, settings.positive_threshold)
    results = engine.lab_compare(liked, k=k, exclude=rated, lam=1.0 - diversity)
    return schemas.LabCompare(
        history=[movie(engine, i) for i in liked[-20:]],
        models={name: recs(engine, items) for name, items in results.items()},
    )


@router.get("/lab/metrics", tags=["model lab"])
def lab_metrics(engine: Engine = Depends(get_engine)) -> dict:
    """Offline experiment results (test set) and the deployed model's own test score."""
    table = engine.offline_results
    return {
        "serving_model": engine.manifest.get("test_metrics", {}),
        "offline": [] if table is None else table.where(table.notna(), None).to_dict(orient="records"),
    }


@router.post("/events", status_code=status.HTTP_202_ACCEPTED, tags=["events"])
def track_event(
    body: schemas.EventIn,
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    db.add(Event(user_id=user, variant=experiments.assign(user, settings), **body.model_dump()))
    db.commit()
    return {"status": "accepted"}
