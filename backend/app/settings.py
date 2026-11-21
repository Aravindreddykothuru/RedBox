"""Configuration from environment variables (prefix REDBOX_) or a backend/.env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_bundle() -> Path:
    """The app's catalogue: MovieLens-32M models (films up to 2023) once exported, else MovieLens-1M."""
    bundles = REPO_ROOT / "data" / "bundles"
    recent = bundles / "ml-32m"
    return recent if (recent / "manifest.json").exists() else bundles / "ml-1m"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDBOX_", env_file=".env", extra="ignore")

    # Local development uses SQLite; production uses Postgres, e.g. Supabase:
    # postgresql+psycopg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'redbox-dev.db').as_posix()}"
    bundle_dir: Path = Field(default_factory=default_bundle)
    # Wikipedia posters, fetched once and kept here at card size (app/posters.py; `python -m app.posters`)
    poster_dir: Path = REPO_ROOT / "data" / "posters"
    # New releases (app/releases.py): how often to look for them (0 = never), how long after startup the
    # first look waits, and whether films found are saved into the bundle (so a restart keeps them)
    releases_refresh_hours: float = 24.0
    releases_first_delay_seconds: float = 60.0
    releases_save: bool = True
    # A host without a permanent disk (a free Hugging Face Space) keeps its SQLite database in a private
    # Hugging Face dataset (app/backup.py): its name ("you/redbox-data"; empty = off) and the interval
    backup_repo: str = ""
    backup_minutes: float = 10.0

    # "local": email + password accounts handled by this API (the default; no third party needed).
    # "supabase": Supabase Auth accounts; the API verifies their access tokens.
    # "dev": no accounts, any caller may act as the user named in an X-User-Id header (tests only).
    auth_mode: Literal["local", "supabase", "dev"] = "local"
    jwt_secret: str | None = None  # local mode: token signing key (random per process if unset)
    token_days: int = 14
    password_scrypt_n: int = 2**15  # scrypt cost; 32 MB of memory per hash
    login_max_failures: int = 10  # per email and per client address ...
    login_window_minutes: int = 15  # ... within this window
    admin_emails: list[str] = []  # local mode: accounts that may read the live metrics
    supabase_url: str | None = None  # new projects: tokens are verified with the project's public JWKS
    supabase_jwt_secret: str | None = None  # legacy projects: HS256 shared secret

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    positive_threshold: float = 4.0  # a rating this high counts as "liked" (same rule as in training)
    log_impressions: bool = True

    # Online A/B test on the "Top picks" row: each user is assigned to one group by a stable hash of
    # (experiment, user id); a group sets the MMR diversity used when the user has not chosen one.
    # Set ab_experiment to "" to switch the test off (everyone gets the first group).
    ab_experiment: str = "topPicksDiversity-1"
    ab_variants: dict[str, float] = {"control": 0.0, "diverse": 0.2}


@lru_cache
def get_settings() -> Settings:
    return Settings()
