"""New releases: found and added while the API runs, warmed up by the app's ratings, shown to every user."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from recsys.data.wikipedia import NEW_ID_OFFSET
from recsys.serving.engine import Engine
from recsys.serving.scorers import IALSScorer
from recsys.serving.testing import make_tiny_bundle

from app import releases as releases_mod
from tests.conftest import make_client

ALICE = {"X-User-Id": "00000000-0000-0000-0000-0000000000a1"}
BEN = {"X-User-Id": "00000000-0000-0000-0000-0000000000b2"}


class FakeEncoder:
    """Stands in for MiniLM: a fixed unit vector per text."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, batch_size: int = 64):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            out[r, sum(map(ord, text)) % self.dim] = 1.0
        return out


def film(q: int, title: str, language: str, links: int, year: int = 2026, genres=("Drama",)) -> dict:
    return {
        "wikidata": f"Q{q}",
        "imdb": None,
        "wiki_title": f"{title} (film)",
        "year": year,
        "genres": list(genres),
        "language": language,
        "links": links,
    }


@pytest.fixture
def app_with_new_films(tmp_path):
    """A tiny bundle with two new films, an iALS generator, and the API on top of it."""
    path = make_tiny_bundle(tmp_path / "tiny")
    base = Engine(path)
    n, dim = base.n_items, base.text_emb.shape[1]
    pd.DataFrame(
        {
            "movie_id": [NEW_ID_OFFSET + 1, NEW_ID_OFFSET + 2],
            "title": ["Old Story", "Other Story"],
            "year": pd.array([2024, 2025], dtype="Int16"),
            "genres": [["Action"], ["Drama"]],
            "popularity": 0,
            "imdb_id": [None, None],
            "poster": [None, None],
            "overview": ["A story.", "Another story."],
            "overview_source": "wikipedia",
            "language": ["English", "Tamil"],
            "links": [30, 4],
            "keywords": [[], []],
        }
    ).to_parquet(path / "new_items.parquet", index=False)
    np.save(path / "new_items_factors.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(path / "new_items_text.npy", np.zeros((2, dim), dtype=np.float32))
    with make_client(path) as c:
        engine = c.app.state.engine
        # an iALS generator for the warm-up: even films lean to factor 0, odd films to factor 1
        engine.scorers["ials"] = IALSScorer(np.tile(np.eye(2), (n // 2 + 1, 1))[:n], alpha=4.0, reg=1.0)
        c.n, c.dim = n, dim
        yield c


def test_a_brand_new_user_gets_the_best_known_releases(app_with_new_films):
    c = app_with_new_films
    out = c.get("/recommendations/new-films", headers=ALICE).json()
    assert out["languages"] == ["English", "Tamil"]
    assert [i["title"] for i in out["items"]] == ["Old Story", "Other Story"]  # 30 Wikipedias before 4
    assert out["items"][0]["reason"] == "New in English: widely talked about"


def test_rating_a_new_film_warms_it_up_for_similar_viewers(app_with_new_films):
    c = app_with_new_films
    engine = c.app.state.engine
    odd = [int(m) for m in engine.items["movie_id"].iloc[[1, 3, 5]]]  # films leaning to factor 1
    for headers in (ALICE, BEN):
        for m in odd:
            c.put(f"/ratings/{m}", json={"rating": 5}, headers=headers)
    first = c.get("/recommendations/new-films", headers=BEN).json()["items"]
    assert first[0]["title"] == "Other Story"  # its story fits factor-1 viewers
    # Alice, who shares Ben's taste, loves the other film: it moves towards their taste
    c.put(f"/ratings/{NEW_ID_OFFSET + 1}", json={"rating": 5}, headers=ALICE)
    after = {i["title"]: i for i in c.get("/recommendations/new-films", headers=BEN).json()["items"]}
    assert after["Old Story"]["reason"] == "New in English: liked by 1 Red Box viewer"
    assert after["Old Story"]["score"] > next(i["score"] for i in first if i["title"] == "Old Story")
    c.delete(f"/ratings/{NEW_ID_OFFSET + 1}", headers=ALICE)  # taking the rating back undoes it
    again = {i["title"]: i for i in c.get("/recommendations/new-films", headers=BEN).json()["items"]}
    assert again["Old Story"]["reason"] == "New in English: its story matches your taste"


def test_the_daily_look_adds_new_releases_without_a_restart(app_with_new_films, monkeypatch):
    c = app_with_new_films
    found = pd.DataFrame(
        [
            film(1, "Old Story", "English", 30, 2024),  # already here: not added twice
            film(777, "Fresh Release", "Hindi", 9),
            film(778, "Somebody Somebody", "Hindi", 9),  # its article is about a person: turned away
        ]
    )
    pages = {
        "Fresh Release (film)": "Fresh Release is a 2026 Indian Hindi-language drama film about a road trip.",
        "Somebody Somebody (film)": "Somebody Somebody (born 1990) is an Indian actor.",
    }
    calls = []

    def fake_wikidata(language, since, until, min_links=0):
        calls.append(language)
        return found[found["language"] == language].reset_index(drop=True)

    def fake_pages(table, cache=None):
        table = table.copy()
        table["summary"] = [pages.get(t) for t in table["wiki_title"]]
        table["poster_url"] = None
        table["title"] = table["wiki_title"].str.replace(" (film)", "", regex=False)
        return table

    monkeypatch.setattr(releases_mod, "wikidata_films", fake_wikidata)
    monkeypatch.setattr(releases_mod, "with_pages", fake_pages)
    releases = c.app.state.releases
    releases.settings = releases.settings.model_copy(update={"releases_save": True})
    assert releases.refresh(encoder=FakeEncoder(c.dim)) == 1
    assert set(calls) == {"English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam"}

    out = c.get("/recommendations/new-films", params={"language": "Hindi"}, headers=ALICE).json()
    assert out["languages"] == ["English", "Tamil", "Hindi"] and out["refreshed_at"]
    assert [i["title"] for i in out["items"]] == ["Fresh Release"]
    assert c.get(f"/movies/{NEW_ID_OFFSET + 777}").json()["title"] == "Fresh Release"
    # saved into the bundle: a restart keeps it
    assert (
        NEW_ID_OFFSET + 777
        in pd.read_parquet(releases.engine.bundle / "new_items.parquet")["movie_id"].tolist()
    )
    # the next look skips what it has and what it turned away this week
    calls.clear()
    assert releases.refresh(encoder=FakeEncoder(c.dim)) == 0
    status = releases.status()
    assert status["added_since_start"] == 1 and status["last_error"] is None


def test_a_failed_look_is_reported_and_the_api_keeps_serving(app_with_new_films, monkeypatch):
    c = app_with_new_films

    def offline(*args, **kwargs):
        raise ConnectionError("no internet")

    monkeypatch.setattr(releases_mod, "wikidata_films", offline)
    assert c.app.state.releases.refresh() == 0
    assert "no internet" in c.app.state.releases.status()["last_error"]
    assert c.get("/recommendations/new-films", headers=ALICE).status_code == 200


def test_releases_status_is_for_operators_only(bundle):
    with make_client(bundle, auth_mode="local", admin_emails=["ops@example.com"]) as c:

        def token(email):
            r = c.post(
                "/auth/register", json={"email": email, "password": "Long-enough-pass-1", "display_name": "x"}
            )
            return {"Authorization": f"Bearer {r.json()['access_token']}"}

        assert c.get("/ops/releases", headers=token("someone@example.com")).status_code == 403
        out = c.get("/ops/releases", headers=token("ops@example.com"))
        assert out.status_code == 200 and out.json()["every_hours"] == 0
