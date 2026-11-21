"""Who is calling? Three modes (REDBOX_AUTH_MODE):

- "local" (default): email + password accounts handled by this API (app/api/accounts.py). Requests carry
  `Authorization: Bearer <token>`; the token's signature, expiry and version are checked on every call,
  and the version must match the account's current one, so revoked tokens stop working at once.
- "supabase": Supabase Auth accounts; the API verifies their access tokens.
- "dev": no accounts; the X-User-Id header names the user (tests and quick experiments only).

User ids are UUIDs in every mode.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app import security
from app.db import Account
from app.services import get_db
from app.settings import Settings, get_settings

DEV_NAMESPACE = uuid.UUID("6f1b2c5e-7c1a-4d5e-9a8b-2f0d3c4b5a69")


@lru_cache
def _jwks_client(supabase_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json")


def _supabase_user(token: str, settings: Settings) -> uuid.UUID:
    try:
        if settings.supabase_jwt_secret:  # legacy projects: shared HS256 secret
            claims = jwt.decode(
                token, settings.supabase_jwt_secret, algorithms=["HS256"], audience="authenticated"
            )
        elif settings.supabase_url:  # current projects: asymmetric keys published as JWKS
            key = _jwks_client(settings.supabase_url).get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["ES256", "RS256"], audience="authenticated")
        else:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "auth is not configured")
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token") from e


def _local_user(token: str, request: Request, db: Session) -> uuid.UUID:
    try:
        user_id, version = security.read_token(token, request.app.state.jwt_secret)
    except (jwt.PyJWTError, KeyError, ValueError) as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token") from e
    account = db.get(Account, user_id)
    if account is None or account.token_version != version:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session ended: please sign in again")
    return user_id


def bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def current_user(
    request: Request, settings: Settings = Depends(get_settings), db: Session = Depends(get_db)
) -> uuid.UUID:
    if settings.auth_mode == "dev":
        raw = request.headers.get("X-User-Id", "dev-user")
        try:
            return uuid.UUID(raw)
        except ValueError:
            return uuid.uuid5(DEV_NAMESPACE, raw)
    token = bearer_token(request)
    if token is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "sign in required", headers={"WWW-Authenticate": "Bearer"}
        )
    if settings.auth_mode == "local":
        return _local_user(token, request, db)
    return _supabase_user(token, settings)


def optional_user(
    request: Request, settings: Settings = Depends(get_settings), db: Session = Depends(get_db)
) -> uuid.UUID | None:
    """For pages anyone may see (movie details): the user if signed in, otherwise None."""
    if settings.auth_mode == "dev" or bearer_token(request):
        return current_user(request, settings, db)
    return None


def admin_user(
    user: uuid.UUID = Depends(current_user),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> uuid.UUID:
    """Operators only (live metrics): accounts listed in REDBOX_ADMIN_EMAILS; anyone in dev mode."""
    if settings.auth_mode == "dev":
        return user
    account = db.get(Account, user)
    admins = {e.strip().lower() for e in settings.admin_emails}
    if account is None or account.email not in admins:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "operators only")
    return user
