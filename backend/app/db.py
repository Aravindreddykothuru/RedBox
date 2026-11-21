"""Database models (SQLAlchemy 2.0). Portable between SQLite (dev, tests) and Postgres (Supabase).

The movie catalog is not stored here: it ships inside the model bundle, so the catalog and the models
always match. The database only holds what users do: profiles, ratings, watchlists, imported watch history and
events.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Uuid, create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Account(Base):
    """Email + password accounts (auth mode "local"). In "supabase" mode, Supabase Auth holds them."""

    __tablename__ = "accounts"
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)  # stored lower-case
    password_hash: Mapped[str] = mapped_column(String(255))
    token_version: Mapped[int] = mapped_column(Integer, default=0)  # +1 revokes all issued tokens
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Profile(Base):
    __tablename__ = "profiles"
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(80))
    favorite_genres: Mapped[list[str]] = mapped_column(JSON, default=list)
    onboarded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Rating(Base):
    __tablename__ = "ratings"
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rating: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)  # sequence order
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class WatchlistItem(Base):
    __tablename__ = "watchlist"
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WatchedItem(Base):
    """Watch history imported from another service (app/history_import.py): implicit feedback, counted as
    liked unless the user rates the film lower. One row per film; times counts rewatches."""

    __tablename__ = "watched"
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    watched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True)
    )  # the latest viewing: sequence order
    source: Mapped[str] = mapped_column(String(20))  # netflix | imdb | letterboxd
    times: Mapped[int] = mapped_column(Integer, default=1)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Event(Base):
    """Impressions and clicks, kept for later analysis (e.g. click-through rate per model)."""

    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    movie_id: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(20))  # impression | click | watchlist | rate
    source_model: Mapped[str | None] = mapped_column(String(32))  # the model or feed row that showed it
    position: Mapped[int | None] = mapped_column(Integer)
    variant: Mapped[str | None] = mapped_column(String(32))  # the user's A/B-test group at the time
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


# Columns added after the first release, created on existing databases at startup (see create_tables).
ADDED_COLUMNS = {"events": {"variant": "VARCHAR(32)"}}


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url in ("sqlite://", "sqlite:///"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory database (tests)
        return create_engine(url, **kwargs)
    return create_engine(url, pool_pre_ping=True)


class Database:
    def __init__(self, url: str):
        self.engine = make_engine(url)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Creates missing tables. On Postgres it also switches row-level security on for every table:
        the API connects as the tables' owner (unaffected), while any other role, such as the public
        REST API a hosted Postgres like Supabase exposes, can read nothing, password hashes included."""
        Base.metadata.create_all(self.engine)
        existing = inspect(self.engine)
        with self.engine.begin() as conn:  # small forward-only migrations: add new nullable columns
            for table, columns in ADDED_COLUMNS.items():
                present = {c["name"] for c in existing.get_columns(table)}
                for name, sql_type in columns.items():
                    if name not in present:
                        conn.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}')
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as conn:
                for table in Base.metadata.sorted_tables:
                    conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY')

    def session(self) -> Iterator[Session]:
        with self.sessions() as s:
            yield s
