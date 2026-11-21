"""Request and response models (they also generate the OpenAPI docs at /docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    popularity: int | None = None
    # a Wikipedia poster: a path on this API ("/posters/<id>.webp", served from its cache); a TMDB poster:
    # a full URL
    poster_url: str | None = None


class MovieDetail(Movie):
    overview: str | None = None
    overview_source: str | None = None  # "tmdb" or "wikipedia"
    language: str | None = None  # new films outside MovieLens (e.g. "Telugu")
    tagline: str | None = None
    runtime: int | None = None
    cast: list[str] = []
    keywords: list[str] = []  # viewer tags from MovieLens (when the bundle has content data)
    avg_rating: float | None = None  # mean MovieLens rating (1-5)
    n_ratings: int | None = None
    tmdb_id: int | None = None
    imdb_id: str | None = None
    your_rating: float | None = None
    in_watchlist: bool = False


class MoviePage(BaseModel):
    total: int
    items: list[Movie]


class Recommendation(Movie):
    score: float
    reason: str


class WhyPart(BaseModel):
    signal: str = Field(description="where the signal comes from, e.g. 'SASRec: the order you watched in'")
    contribution: float = Field(description="its exact share of the ranker's score (TreeSHAP)")
    share: float = Field(description="contribution / sum of |contributions|: signed, -1 to 1")


class WhyOut(BaseModel):
    """How the hybrid's ranker scored one film for you."""

    available: bool = Field(description="false when the film is not among your candidates")
    rank: int | None = None
    candidates: int | None = None
    proposed_by: list[str] = []
    parts: list[WhyPart] = []


class NewFilmsOut(BaseModel):
    languages: list[str] = Field(description="every language with new films, e.g. English, Hindi, Tamil")
    language: str | None = Field(None, description="the one language shown, or none for all of them")
    refreshed_at: datetime | None = Field(None, description="when the API last looked for new releases")
    items: list[Recommendation]


class ReleasesStatus(BaseModel):
    new_films: int = Field(description="films newer than the models' data, served from their story")
    languages: list[str]
    refreshed_at: datetime | None = Field(None, description="the last look for new releases")
    last_added: int = Field(description="films the last look added")
    added_since_start: int
    films_with_ratings: int = Field(description="new films that app users have rated (warm-up)")
    every_hours: float = Field(description="how often it looks (0: never)")
    last_error: str | None = None


class FeedRow(BaseModel):
    key: str
    title: str
    items: list[Recommendation]


class Feed(BaseModel):
    rows: list[FeedRow]


class RatingIn(BaseModel):
    rating: float = Field(ge=0.5, le=5.0, description="0.5 to 5 stars in half-star steps")

    @field_validator("rating")
    @classmethod
    def half_stars(cls, v: float) -> float:
        if round(v * 2) != v * 2:
            raise ValueError("ratings go in half-star steps")
        return v


class RatedMovie(Movie):
    rating: float
    rated_at: datetime


class OnboardingRating(RatingIn):
    movie_id: int


class OnboardingIn(BaseModel):
    favorite_genres: list[str] = []
    ratings: list[OnboardingRating] = []
    display_name: str | None = Field(default=None, max_length=80)


class ProfileOut(BaseModel):
    display_name: str | None
    favorite_genres: list[str]
    onboarded: bool
    n_ratings: int
    n_watched: int = 0  # films from an imported watch history


class EventIn(BaseModel):
    movie_id: int
    type: Literal["impression", "click", "watchlist", "rate"]
    source_model: str | None = Field(default=None, max_length=32)
    position: int | None = Field(default=None, ge=0)


class LabCompare(BaseModel):
    history: list[Movie]
    models: dict[str, list[Recommendation]]


class LabUser(BaseModel):
    user_id: int
    n_liked: int


class Health(BaseModel):
    status: str
    dataset: str
    bundle_created: str
    n_items: int
    models: list[str]
    semantic_search: bool = False
    serving_test_metrics: dict[str, float]
