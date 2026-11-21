"""Accounts (auth mode "local"): sign up, sign in, password change, sign out everywhere; plus what every
user may do in any mode: edit the profile, download all their data, delete the account."""

from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import security
from app.auth import current_user
from app.db import Account, Event, Profile, Rating, WatchedItem, WatchlistItem, now
from app.services import get_db, update_new_films_later
from app.settings import Settings, get_settings

router = APIRouter(tags=["accounts"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email(v: str) -> str:
    v = v.strip().lower()
    if len(v) > 254 or not _EMAIL.match(v):
        raise ValueError("enter a valid email address")
    return v


def _password(v: str) -> str:
    if len(v) < 8:
        raise ValueError("use at least 8 characters")
    if not v.strip():
        raise ValueError("the password cannot be only spaces")
    return v


class RegisterIn(BaseModel):
    email: str
    password: str = Field(max_length=128)
    display_name: str | None = Field(default=None, max_length=80)

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        return _email(v)

    @field_validator("password")
    @classmethod
    def strong_enough(cls, v: str) -> str:
        return _password(v)


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class PasswordChangeIn(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: str = Field(max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong_enough(cls, v: str) -> str:
        return _password(v)


class PasswordConfirmIn(BaseModel):
    password: str = Field(default="", max_length=128)


class AccountOut(BaseModel):
    user_id: uuid.UUID
    email: str | None
    display_name: str | None
    is_admin: bool = False  # may read the live metrics (REDBOX_ADMIN_EMAILS)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AccountOut


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    favorite_genres: list[str] | None = Field(default=None, max_length=30)


def _is_admin(email: str, settings: Settings) -> bool:
    return email in {e.strip().lower() for e in settings.admin_emails}


def _local_only(settings: Settings) -> None:
    if settings.auth_mode != "local":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "accounts are managed elsewhere in this deployment")


def _token(request: Request, account: Account, settings: Settings, db: Session) -> TokenOut:
    profile = db.get(Profile, account.user_id)
    return TokenOut(
        access_token=security.issue_token(
            account.user_id, account.token_version, request.app.state.jwt_secret, settings.token_days
        ),
        expires_in=settings.token_days * 86400,
        user=AccountOut(
            user_id=account.user_id,
            email=account.email,
            display_name=profile.display_name if profile else None,
            is_admin=_is_admin(account.email, settings),
        ),
    )


@router.post("/auth/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterIn,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenOut:
    _local_only(settings)
    account = Account(
        user_id=uuid.uuid4(),
        email=body.email,
        password_hash=security.hash_password(body.password, n=settings.password_scrypt_n),
        last_login_at=now(),
    )
    db.add(account)
    db.add(Profile(user_id=account.user_id, display_name=(body.display_name or "").strip() or None))
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists") from e
    return _token(request, account, settings, db)


@router.post("/auth/login", response_model=TokenOut)
def login(
    body: LoginIn, request: Request, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
) -> TokenOut:
    _local_only(settings)
    email = body.email.strip().lower()
    client = request.client.host if request.client else "unknown"
    keys = (f"email:{email}", f"ip:{client}")
    limiter: security.RateLimiter = request.app.state.login_limiter
    if limiter.blocked(*keys):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"too many failed attempts; try again in {settings.login_window_minutes} minutes",
        )
    account = db.scalar(select(Account).where(Account.email == email))
    if account is None:
        security.dummy_verify(body.password, settings.password_scrypt_n)  # same timing as a real check
        ok = False
    else:
        ok = security.verify_password(body.password, account.password_hash)
    if not ok:
        limiter.fail(*keys)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong email or password")
    limiter.reset(f"email:{email}")
    account.last_login_at = now()
    db.commit()
    return _token(request, account, settings, db)


@router.get("/auth/me", response_model=AccountOut)
def me(
    user: uuid.UUID = Depends(current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AccountOut:
    account = db.get(Account, user)
    profile = db.get(Profile, user)
    return AccountOut(
        user_id=user,
        email=account.email if account else None,
        display_name=profile.display_name if profile else None,
        is_admin=settings.auth_mode == "dev" or (account is not None and _is_admin(account.email, settings)),
    )


@router.post("/auth/password", response_model=TokenOut)
def change_password(
    body: PasswordChangeIn,
    request: Request,
    user: uuid.UUID = Depends(current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenOut:
    """Changes the password and signs out every other device (all earlier tokens stop working)."""
    _local_only(settings)
    account = db.get(Account, user)
    if account is None or not security.verify_password(body.current_password, account.password_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "the current password is wrong")
    account.password_hash = security.hash_password(body.new_password, n=settings.password_scrypt_n)
    account.token_version += 1
    db.commit()
    return _token(request, account, settings, db)


@router.post("/auth/logout-all", status_code=status.HTTP_204_NO_CONTENT)
def logout_everywhere(
    user: uuid.UUID = Depends(current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    _local_only(settings)
    account = db.get(Account, user)
    if account is not None:
        account.token_version += 1
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/me/profile", status_code=status.HTTP_204_NO_CONTENT, tags=["me"])
def update_profile(
    body: ProfileUpdate, user: uuid.UUID = Depends(current_user), db: Session = Depends(get_db)
) -> Response:
    profile = db.get(Profile, user) or Profile(user_id=user)
    if body.display_name is not None:
        profile.display_name = body.display_name.strip() or None
    if body.favorite_genres is not None:
        profile.favorite_genres = body.favorite_genres
    db.add(profile)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/export", tags=["me"])
def export_my_data(user: uuid.UUID = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    """Everything stored about the signed-in user, as one JSON document."""
    account = db.get(Account, user)
    profile = db.get(Profile, user)
    ratings = db.scalars(select(Rating).where(Rating.user_id == user).order_by(Rating.created_at)).all()
    watchlist = db.scalars(select(WatchlistItem).where(WatchlistItem.user_id == user)).all()
    watched = db.scalars(
        select(WatchedItem).where(WatchedItem.user_id == user).order_by(WatchedItem.watched_at)
    ).all()
    events = db.scalars(select(Event).where(Event.user_id == user).order_by(Event.created_at)).all()
    return {
        "user_id": str(user),
        "account": None
        if account is None
        else {
            "email": account.email,
            "created_at": account.created_at,
            "last_login_at": account.last_login_at,
        },
        "profile": None
        if profile is None
        else {
            "display_name": profile.display_name,
            "favorite_genres": profile.favorite_genres,
            "onboarded_at": profile.onboarded_at,
        },
        "ratings": [{"movie_id": r.movie_id, "rating": r.rating, "rated_at": r.created_at} for r in ratings],
        "watchlist": [{"movie_id": w.movie_id, "added_at": w.added_at} for w in watchlist],
        "imported_history": [
            {"movie_id": w.movie_id, "watched_at": w.watched_at, "source": w.source, "times": w.times}
            for w in watched
        ],
        "events": [
            {
                "movie_id": e.movie_id,
                "type": e.type,
                "source": e.source_model,
                "position": e.position,
                "at": e.created_at,
            }
            for e in events
        ],
    }


@router.post("/me/delete", status_code=status.HTTP_204_NO_CONTENT, tags=["me"])
def delete_my_account(
    body: PasswordConfirmIn,
    request: Request,
    background: BackgroundTasks,
    user: uuid.UUID = Depends(current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Deletes the account and everything stored about it. Local accounts confirm with the password."""
    account = db.get(Account, user)
    if settings.auth_mode == "local" and (
        account is None or not security.verify_password(body.password, account.password_hash)
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "the password is wrong")
    for model in (Event, Rating, WatchlistItem, WatchedItem, Profile, Account):
        db.execute(delete(model).where(model.user_id == user))
    db.commit()
    update_new_films_later(request, background)  # their ratings of new films no longer count
    return Response(status_code=status.HTTP_204_NO_CONTENT)
