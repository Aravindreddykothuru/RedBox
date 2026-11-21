"""Watch together: films for a group, from every member's hybrid scores combined by a group strategy."""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from recsys.serving.engine import Engine
from sqlalchemy.orm import Session

from app import schemas
from app.auth import current_user
from app.services import get_db, get_engine, movie, user_history
from app.settings import Settings, get_settings

router = APIRouter(tags=["group"])

Strategy = Literal["average", "least_misery", "most_pleasure"]


class Friend(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    liked: list[int] = Field(default=[], max_length=50, description="movie ids of films they love")


class GroupIn(BaseModel):
    include_me: bool = True
    friends: list[Friend] = Field(default=[], max_length=5)
    strategy: Strategy = "average"
    k: int = Field(10, ge=1, le=30)


class GroupPick(BaseModel):
    movie: schemas.Movie
    score: float = Field(description="the group's score under the strategy, 0-1")
    fits: list[float] = Field(description="how well it fits each member, 0-1, in the order of `members`")


class GroupOut(BaseModel):
    members: list[str]
    strategy: Strategy
    items: list[GroupPick]


@router.post("/group", response_model=GroupOut)
def watch_together(
    body: GroupIn,
    engine: Engine = Depends(get_engine),
    db: Session = Depends(get_db),
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> GroupOut:
    """Films for several people at once. The members are the signed-in user (their ratings and imported
    history) and friends described by a few films they love. The hybrid scores one shared pool of
    candidates for every member; the strategy combines the fits: average (most satisfaction overall),
    least misery (nobody is unhappy) or most pleasure (someone's favourite). Films the signed-in user has
    rated or watched, and the friends' films, are never proposed."""
    histories: list[list[int]] = []
    names: list[str] = []
    exclude: set[int] = set()
    if body.include_me:
        liked, seen = user_history(db, engine, user, settings.positive_threshold)
        histories.append(liked)
        names.append("You")
        exclude |= {i for i in seen if i < engine.n_items}
    for f in body.friends:
        idx = engine.movie_to_idx
        histories.append([idx[m] for m in dict.fromkeys(f.liked) if m in idx and idx[m] < engine.n_items])
        names.append(f.name.strip())
    if len(histories) < 2:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "add at least one friend to watch with")
    picks = engine.group(histories, body.strategy, k=body.k, exclude=exclude)
    return GroupOut(
        members=names,
        strategy=body.strategy,
        items=[GroupPick(movie=movie(engine, i), score=score, fits=fits) for i, score, fits in picks],
    )
