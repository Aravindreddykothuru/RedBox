"""Viewer keywords for each movie, from the free-text tags in MovieLens 32M.

MovieLens users tag movies with words such as "mind-bending", "time travel" or "dystopia". Those tags
describe what a film is about far better than its genres, and they are free (no API key), so they are
the main ingredient of the movie texts behind semantic search. MovieLens movie ids are stable across
releases, so ML-32M tags attach directly to ML-1M movies.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from recsys.data.registry import get_spec
from recsys.paths import raw_dir

# Tags that describe the viewer's situation rather than the film.
STOP_TAGS = {
    "seen", "seen it", "seen more than once", "watched", "to see", "want to see", "must see", "own",
    "owned", "own it", "dvd", "on dvd", "dvd-video", "bd-r", "bd-video", "clv", "vhs", "netflix",
    "netflix queue", "in netflix queue", "tivo", "imdb top 250", "favorite", "favorites", "my favorite",
    "fav", "boring", "overrated", "underrated", "not seen", "library", "watchlist",
    "erlend's dvds", "ei muista", "betamax", "r", "pg-13", "pg", "nr",
}  # fmt: skip
_SPACES = re.compile(r"\s+")


def tags_csv() -> Path:
    """tags.csv of ML-32M, extracted on first use from the already-verified archive."""
    spec = get_spec("ml-32m")
    target = raw_dir(spec.extracted_dir) / "tags.csv"
    if not target.exists():
        archive = raw_dir() / spec.archive_name
        if not archive.exists():
            raise FileNotFoundError(f"{archive} missing: run `recsys download ml-32m` first")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf, zf.open(f"{spec.extracted_dir}/tags.csv") as src:
            tmp = target.with_suffix(".part")
            tmp.write_bytes(src.read())
            tmp.replace(target)
    return target


def normalise_tag(tag: str) -> str:
    return _SPACES.sub(" ", str(tag).strip().lower())


def movie_keywords(
    tags: pd.DataFrame, movie_ids, per_movie: int = 12, min_users: int = 2
) -> dict[int, list[str]]:
    """Top keywords per movie, ranked by how many different users applied them.

    ``tags`` has columns userId, movieId, tag. A keyword must be used by at least ``min_users`` people,
    which removes one-off private notes; movies with no such keyword fall back to their most-used tags.
    """
    wanted = set(int(m) for m in movie_ids)
    t = tags[tags["movieId"].isin(wanted)].copy()
    t["tag"] = t["tag"].map(normalise_tag)
    t = t[(t["tag"].str.len().between(2, 40)) & ~t["tag"].isin(STOP_TAGS)]
    counts = (
        t.drop_duplicates(["userId", "movieId", "tag"])
        .groupby(["movieId", "tag"])
        .size()
        .rename("users")
        .reset_index()
        .sort_values(["movieId", "users", "tag"], ascending=[True, False, True], kind="stable")
    )
    out: dict[int, list[str]] = {}
    for movie, group in counts.groupby("movieId", sort=False):
        strong = group[group["users"] >= min_users]["tag"].tolist()
        out[int(movie)] = (strong or group["tag"].tolist()[:3])[:per_movie]
    return out


def load_keywords(movie_ids, per_movie: int = 12) -> dict[int, list[str]]:
    tags = pd.read_csv(tags_csv(), usecols=["userId", "movieId", "tag"], dtype={"tag": "string"})
    tags = tags.dropna(subset=["tag"])
    return movie_keywords(tags, movie_ids, per_movie=per_movie)
