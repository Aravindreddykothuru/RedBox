"""Accounts without a third party: password hashing, signed login tokens and login rate limiting.

- Passwords: scrypt (memory-hard; Python standard library), a random 16-byte salt per password, and a
  constant-time comparison. The stored string carries its own parameters, so the cost can be raised
  later without breaking existing accounts.
- Tokens: JWT (HS256) signed with REDBOX_JWT_SECRET, carrying the account's token version, so
  "sign out everywhere" and a password change revoke every token issued before.
- Rate limiting: at most N failed attempts per email and per client address in a sliding window.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time
import uuid
from collections import defaultdict, deque

import jwt

ISSUER = "redbox"
AUDIENCE = "redbox-app"


# ------------------------------------------------------------------------------------------ passwords
def hash_password(password: str, n: int = 2**15, r: int = 8, p: int = 1) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=256 * 1024 * 1024, dklen=32)
    b64 = base64.b64encode
    return f"scrypt${n}${r}${p}${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=256 * 1024 * 1024,
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# A hash to compare against when the email is unknown, so a login takes the same time whether or not the
# account exists (no account enumeration by timing).
_DUMMY_HASH: dict[int, str] = {}


def dummy_verify(password: str, n: int) -> None:
    if n not in _DUMMY_HASH:
        _DUMMY_HASH[n] = hash_password("not-a-real-password", n=n)
    verify_password(password, _DUMMY_HASH[n])


# ------------------------------------------------------------------------------------------ tokens
def issue_token(user_id: uuid.UUID, token_version: int, secret: str, days: int) -> str:
    now = int(time.time())
    claims = {
        "sub": str(user_id),
        "ver": token_version,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + days * 86400,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def read_token(token: str, secret: str) -> tuple[uuid.UUID, int]:
    """(user id, token version); raises jwt.PyJWTError or ValueError on anything invalid or expired."""
    claims = jwt.decode(token, secret, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER)
    return uuid.UUID(claims["sub"]), int(claims["ver"])


# ------------------------------------------------------------------------------------------ rate limit
class RateLimiter:
    """Sliding-window counter of failures per key (in memory: one API process, as on the free tier)."""

    def __init__(self, max_failures: int, window_seconds: float):
        self.max_failures = max_failures
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        return hits

    def blocked(self, *keys: str) -> bool:
        now = time.monotonic()
        with self._lock:
            return any(len(self._trim(k, now)) >= self.max_failures for k in keys)

    def fail(self, *keys: str) -> None:
        now = time.monotonic()
        with self._lock:
            for k in keys:
                self._trim(k, now).append(now)

    def reset(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._hits.pop(k, None)
