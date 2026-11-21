"""Posters: fetched once from the source, shrunk to card size, then served from the API's disk cache."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app import posters
from tests.conftest import make_client


def png(width: int, height: int, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), (200, 30, 40) if mode == "RGB" else (200, 30, 40, 128)).save(out, "PNG")
    return out.getvalue()


def test_to_webp_shrinks_to_card_size_and_keeps_proportions():
    img = Image.open(io.BytesIO(posters.to_webp(png(684, 1026))))
    assert img.format == "WEBP" and img.size == (342, 513)
    small = Image.open(io.BytesIO(posters.to_webp(png(220, 330))))
    assert small.size == (220, 330)  # never enlarged
    assert Image.open(io.BytesIO(posters.to_webp(png(300, 450, "RGBA")))).mode == "RGB"
    with pytest.raises(posters.PosterUnavailable):
        posters.to_webp(b"<html>not an image</html>")


@pytest.fixture
def poster_client(bundle, tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_get(url: str, max_wait: float = 0.0) -> bytes:
        calls.append(url)
        if "missing" in url:
            raise posters.PosterUnavailable("throttled")
        return png(684, 1026)

    monkeypatch.setattr(posters, "_get", fake_get)
    with make_client(bundle, poster_dir=tmp_path / "posters") as c:
        engine = c.app.state.engine
        engine.items["poster"] = None
        engine.items.loc[0, "poster"] = "https://upload.example.org/a.jpg"
        engine.items.loc[1, "poster"] = "https://upload.example.org/missing.jpg"
        c.calls = calls
        c.ids = [int(engine.items["movie_id"].iloc[i]) for i in range(3)]
        yield c


def get(c, movie_id: int):
    return c.get(f"/posters/{movie_id}.webp", follow_redirects=False)


def test_uncached_posters_redirect_while_the_api_caches_them(poster_client):
    c = poster_client
    first = get(c, c.ids[0])
    assert first.status_code == 307 and first.headers["location"] == "https://upload.example.org/a.jpg"
    assert first.headers["cache-control"] == "no-store"  # so the next view asks the API again
    posters._queue.join()  # the background cacher has saved it
    cached = get(c, c.ids[0])
    assert cached.status_code == 200 and cached.headers["content-type"] == "image/webp"
    assert "immutable" in cached.headers["cache-control"]
    assert Image.open(io.BytesIO(cached.content)).size == (342, 513)
    assert get(c, c.ids[0]).content == cached.content and len(c.calls) == 1  # fetched from the source once


def test_poster_errors(poster_client):
    c = poster_client
    assert get(c, c.ids[1]).status_code == 307  # the source refuses the API: the browser still tries it
    posters._queue.join()
    assert get(c, c.ids[1]).status_code == 307  # nothing cached, asked again next time
    assert get(c, c.ids[2]).status_code == 404  # this film has no poster
    assert get(c, 999999999).status_code == 404  # no such film


def test_films_point_at_the_api_s_poster_cache(poster_client):
    c = poster_client
    out = c.get(f"/movies/{c.ids[0]}").json()
    assert out["poster_url"] == f"/posters/{c.ids[0]}.webp"
    assert c.get(f"/movies/{c.ids[2]}").json()["poster_url"] is None


def test_warm_order_alternates_the_most_watched_with_new_films():
    import pandas as pd

    films = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 1_000_000_001, 1_000_000_002],
            "poster": ["a", "b", "c", "n1", "n2"],
            "popularity": [10, 30, 20, 0, 0],
        }
    )
    assert [m for m, _ in posters.warm_order(films)] == [2, 1_000_000_001, 3, 1_000_000_002, 1]


def test_pacing_slows_down_on_429_and_gives_up_after_its_budget(monkeypatch):
    replies: list[int] = []

    class Reply:
        def __init__(self, status: int):
            self.status_code, self.headers, self.content = status, {"retry-after": "1"}, b"image"

    class Session:
        def get(self, url, timeout):
            return Reply(replies.pop(0))

    slept: list[float] = []
    monkeypatch.setattr(posters, "_session", Session())
    monkeypatch.setattr(posters.time, "sleep", slept.append)
    monkeypatch.setattr(posters, "_pace", posters.MIN_INTERVAL)

    replies[:] = [429, 429, 200]
    assert posters._get("u", max_wait=10) == b"image"
    assert posters._pace > posters.MIN_INTERVAL  # slower after two 429s, eased a little by the success

    replies[:] = [429] * 50
    with pytest.raises(posters.PosterUnavailable):
        posters._get("u", max_wait=6.0)
    assert sum(s for s in slept[-12:] if s == 1.0) <= 6.0  # Retry-After waits stay within the budget
    assert posters._pace <= posters.MAX_INTERVAL
