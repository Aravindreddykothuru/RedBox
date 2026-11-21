from __future__ import annotations

import pandas as pd
import pytest

from recsys.data import tmdb


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Movie 11 exists (first answer is a rate-limit), movie 22 is unknown to TMDB."""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/22"):
            return FakeResponse(404)
        if self.calls.count(url) == 1:
            return FakeResponse(429, headers={"Retry-After": "0"})
        return FakeResponse(
            200,
            {
                "poster_path": "/p.jpg",
                "overview": "A test movie.",
                "runtime": 101,
                "credits": {"cast": [{"name": f"Actor {i}"} for i in range(8)]},
            },
        )


def test_enrich_uses_cache_and_handles_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("TMDB_READ_TOKEN", "test-token")
    links = pd.DataFrame({"item_id": [1, 2], "tmdb_id": [11, 22], "imdb_id": ["0114709", "0113497"]})
    session = FakeSession()
    out = tmdb.enrich(links, tmp_path / "cache", session=session, delay=0)
    assert out["item_id"].tolist() == [1]  # unknown movie skipped
    row = out.iloc[0]
    assert row["poster_path"] == "/p.jpg" and len(row["cast"]) == 5 and row["runtime"] == 101
    calls = len(session.calls)
    tmdb.enrich(links, tmp_path / "cache", session=session, delay=0)  # second run: everything cached
    assert len(session.calls) == calls


def test_missing_credentials_are_reported(monkeypatch):
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TMDB_READ_TOKEN"):
        tmdb.enrich(pd.DataFrame(columns=["item_id", "tmdb_id", "imdb_id"]), __import__("pathlib").Path("."))


def test_poster_url():
    assert tmdb.poster_url("/x.jpg") == "https://image.tmdb.org/t/p/w342/x.jpg"
    assert tmdb.poster_url(None) is None
