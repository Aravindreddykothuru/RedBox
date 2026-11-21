"""Movie metadata from TMDB (posters, plot overviews, cast) for the app.

Needs a free TMDB account. Put the key in an environment variable, never in code:
    TMDB_READ_TOKEN  (API Read Access Token, sent as a Bearer header)   - preferred
    TMDB_API_KEY     (v3 API key, sent as a query parameter)

MovieLens ids map to TMDB ids through MovieLens' links.csv (ids are stable across MovieLens releases).
Each movie is fetched once and cached as JSON, so re-runs are free and resumable.
Attribution required by TMDB: "This product uses the TMDB API but is not endorsed or certified by TMDB."
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"


def load_links(links_csv: Path) -> pd.DataFrame:
    """movieId -> tmdbId / imdbId from a MovieLens links.csv."""
    links = pd.read_csv(links_csv, dtype={"imdbId": str})
    links = links.dropna(subset=["tmdbId"]).astype({"tmdbId": "int64"})
    return links.rename(columns={"movieId": "item_id", "tmdbId": "tmdb_id", "imdbId": "imdb_id"})


def _auth() -> tuple[dict, dict]:
    token, key = os.environ.get("TMDB_READ_TOKEN"), os.environ.get("TMDB_API_KEY")
    if token:
        return {"Authorization": f"Bearer {token}", "accept": "application/json"}, {}
    if key:
        return {"accept": "application/json"}, {"api_key": key}
    raise RuntimeError("Set TMDB_READ_TOKEN or TMDB_API_KEY (free at themoviedb.org -> Settings -> API)")


def fetch_movie(
    session: requests.Session, tmdb_id: int, headers: dict, params: dict, retries: int = 5
) -> dict | None:
    """One movie with its top-billed cast. None if TMDB does not know the id."""
    url = f"{API}/movie/{tmdb_id}"
    for attempt in range(retries):
        r = session.get(url, headers=headers, params={**params, "append_to_response": "credits"}, timeout=30)
        if r.status_code == 404:
            return None
        if r.status_code == 429:  # rate limited: wait as told, then retry
            time.sleep(float(r.headers.get("Retry-After", 2**attempt)))
            continue
        r.raise_for_status()
        d = r.json()
        return {
            "tmdb_id": tmdb_id,
            "poster_path": d.get("poster_path"),
            "backdrop_path": d.get("backdrop_path"),
            "overview": d.get("overview") or None,
            "tagline": d.get("tagline") or None,
            "runtime": d.get("runtime"),
            "release_date": d.get("release_date") or None,
            "original_language": d.get("original_language"),
            "cast": [c["name"] for c in (d.get("credits") or {}).get("cast", [])[:5]],
        }
    raise RuntimeError(f"TMDB kept rate-limiting movie {tmdb_id}")


def enrich(
    links: pd.DataFrame, cache_dir: Path, *, session: requests.Session | None = None, delay: float = 0.05
) -> pd.DataFrame:
    """Metadata for every linked movie, using and filling the per-movie JSON cache."""
    headers, params = _auth()
    session = session or requests.Session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_id, tmdb_id, imdb_id in links[["item_id", "tmdb_id", "imdb_id"]].itertuples(index=False):
        path = cache_dir / f"{tmdb_id}.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
        else:
            meta = fetch_movie(session, int(tmdb_id), headers, params)
            path.write_text(json.dumps(meta), encoding="utf-8")
            time.sleep(delay)  # stay well under TMDB's rate limit
        if meta:
            rows.append({"item_id": int(item_id), "imdb_id": imdb_id, **meta})
    return pd.DataFrame(rows)


def poster_url(poster_path: str | None, size: str = "w342") -> str | None:
    return f"{IMAGE_BASE}/{size}{poster_path}" if poster_path else None
