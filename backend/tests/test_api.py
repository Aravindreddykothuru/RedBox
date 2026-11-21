"""API tests on the tiny bundle: items 0-9 form taste group 1, items 10-19 taste group 2.
Movie ids are item index + 100."""

from __future__ import annotations

import time
import uuid

import jwt

from tests.conftest import make_client

ALICE = {"X-User-Id": "alice"}
BOB = {"X-User-Id": "bob"}


def rate(client, movie_id, stars, who=ALICE):
    r = client.put(f"/ratings/{movie_id}", json={"rating": stars}, headers=who)
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_items"] == 20
    assert set(body["models"]) == {"hybrid", "ease", "markov", "mostpop"}


def test_catalog_browse_search_and_detail(client):
    page = client.get("/movies", params={"limit": 5}).json()
    assert page["total"] == 20 and len(page["items"]) == 5
    pops = [m["popularity"] for m in page["items"]]
    assert pops == sorted(pops, reverse=True)  # most popular first
    assert client.get("/movies", params={"q": "movie 1"}).json()["total"] == 11  # "Movie 1", "Movie 10".."19"
    assert all(
        "Action" in m["genres"] for m in client.get("/movies", params={"genre": "Action"}).json()["items"]
    )
    assert client.get("/movies/genres").json() == ["Action", "Comedy", "Drama"]
    assert client.get("/movies/105").json()["title"] == "Movie 5"
    assert client.get("/movies/999").status_code == 404
    assert len(client.get("/movies/100/similar", params={"k": 4}).json()) == 4


def test_ratings_drive_recommendations_in_real_time(client):
    # Alice likes taste group 2 (items 10-19 -> movie ids 110-119)
    for m in (110, 111, 112, 113):
        rate(client, m, 5)
    rate(client, 114, 1.5)  # rated but disliked
    recs = client.get("/recommendations", params={"k": 5}, headers=ALICE).json()
    ids = {r["movie_id"] for r in recs}
    assert not ids & {110, 111, 112, 113, 114}  # nothing she already rated, liked or not
    assert sum(110 <= i <= 119 for i in ids) >= 4  # mostly from her taste group
    assert all(r["reason"] for r in recs)
    # a new rating changes the next response immediately (no retraining)
    rate(client, recs[0]["movie_id"], 4.5)
    after = {r["movie_id"] for r in client.get("/recommendations", params={"k": 5}, headers=ALICE).json()}
    assert recs[0]["movie_id"] not in after
    mine = client.get("/me/ratings", headers=ALICE).json()
    assert len(mine) == 6 and mine[0]["movie_id"] == recs[0]["movie_id"]  # newest first


def test_rating_validation_and_delete(client):
    assert client.put("/ratings/100", json={"rating": 6}, headers=ALICE).status_code == 422
    assert client.put("/ratings/100", json={"rating": 3.3}, headers=ALICE).status_code == 422
    assert client.put("/ratings/999", json={"rating": 4}, headers=ALICE).status_code == 404
    rate(client, 100, 4)
    assert client.get("/movies/100", headers=ALICE).json()["your_rating"] == 4
    assert client.delete("/ratings/100", headers=ALICE).status_code == 204
    assert client.get("/movies/100", headers=ALICE).json()["your_rating"] is None


def test_users_are_isolated(client):
    rate(client, 101, 5, who=ALICE)
    assert client.get("/me/ratings", headers=BOB).json() == []
    assert client.get("/movies/101", headers=BOB).json()["your_rating"] is None


def test_onboarding_flow(client):
    seeds = client.get("/onboarding/seeds", params={"n": 5}, headers=ALICE).json()
    assert len(seeds) == 5
    body = {
        "favorite_genres": ["Drama"],
        "display_name": "Alice",
        "ratings": [{"movie_id": s["movie_id"], "rating": 5} for s in seeds[:3]],
    }
    first = client.post("/onboarding", json=body, headers=ALICE).json()
    assert first and not {r["movie_id"] for r in first} & {s["movie_id"] for s in seeds[:3]}
    profile = client.get("/me/profile", headers=ALICE).json()
    assert profile == {
        "display_name": "Alice",
        "favorite_genres": ["Drama"],
        "onboarded": True,
        "n_ratings": 3,
        "n_watched": 0,
    }


def test_feed_new_user_vs_active_user(client):
    keys = [row["key"] for row in client.get("/recommendations/feed", headers=BOB).json()["rows"]]
    assert keys == ["trending"]  # nothing liked yet
    for m in (100, 101, 102):
        rate(client, m, 5)
    keys = [row["key"] for row in client.get("/recommendations/feed", headers=ALICE).json()["rows"]]
    assert keys[0] == "top_picks" and "because_you_liked" in keys and keys[-1] == "trending"


def test_model_lab_and_metrics(client):
    users = client.get("/lab/users", params={"limit": 3}).json()
    assert len(users) == 3 and users[0]["n_liked"] > 0
    lab = client.get(
        "/lab/compare", params={"dataset_user": users[0]["user_id"], "k": 3}, headers=ALICE
    ).json()
    assert set(lab["models"]) == {"hybrid", "ease", "markov", "mostpop"} and lab["history"]
    assert client.get("/lab/compare", params={"dataset_user": 99999}, headers=ALICE).status_code == 404
    assert client.get("/lab/metrics").json()["serving_model"] == {"ndcg@10": 0.5}


def test_watchlist_events_and_bad_model(client):
    assert client.put("/watchlist/103", headers=ALICE).status_code == 204
    assert [m["movie_id"] for m in client.get("/me/watchlist", headers=ALICE).json()] == [103]
    assert client.get("/movies/103", headers=ALICE).json()["in_watchlist"] is True
    assert client.delete("/watchlist/103", headers=ALICE).status_code == 204
    event = {"movie_id": 103, "type": "click", "source_model": "hybrid", "position": 0}
    assert client.post("/events", json=event, headers=ALICE).status_code == 202
    assert client.get("/recommendations", params={"model": "nope"}, headers=ALICE).status_code == 422


def test_supabase_tokens(bundle):
    secret = "test-secret-at-least-32-characters-long!!"
    with make_client(bundle, auth_mode="supabase", supabase_jwt_secret=secret) as client:
        assert client.get("/me/ratings").status_code == 401  # no token
        bad = jwt.encode(
            {"sub": str(uuid.uuid4()), "aud": "authenticated"}, "wrong-secret-wrong-secret-wrong!!!", "HS256"
        )
        assert client.get("/me/ratings", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
        user = uuid.uuid4()
        token = jwt.encode(
            {"sub": str(user), "aud": "authenticated", "exp": time.time() + 60}, secret, "HS256"
        )
        auth = {"Authorization": f"Bearer {token}"}
        assert client.put("/ratings/100", json={"rating": 5}, headers=auth).status_code == 200
        assert client.get("/me/ratings", headers=auth).json()[0]["movie_id"] == 100
        assert client.get("/movies/100").status_code == 200  # browsing needs no sign-in


def test_semantic_search_and_content_similarity(client):
    from recsys.serving.testing import KeywordEncoder

    assert client.get("/health").json()["semantic_search"] is False  # tiny bundle ships no ONNX model
    assert client.get("/search/semantic", params={"q": "comedy"}).status_code == 503
    client.app.state.engine.encoder = KeywordEncoder()  # stand-in for the sentence encoder
    assert client.get("/health").json()["semantic_search"] is True
    hits = client.get("/search/semantic", params={"q": "something with drama", "k": 4}).json()
    assert len(hits) == 4 and all(h["genres"] == ["Drama"] for h in hits)
    assert hits[0]["reason"].startswith("Viewers tag it: drama")
    assert client.get("/search/semantic", params={"q": "x"}).status_code == 422  # too short
    similar = client.get("/movies/101/similar", params={"mode": "content", "k": 3}).json()
    assert [m["genres"] for m in similar] == [["Comedy"]] * 3
    detail = client.get("/movies/101").json()
    assert detail["keywords"] == ["comedy", "tag1"]


def test_model_lab_diversity_changes_only_the_hybrid(client):
    params = {"dataset_user": 1, "k": 5}
    plain = client.get("/lab/compare", params=params, headers=ALICE).json()["models"]
    diverse = client.get("/lab/compare", params={**params, "diversity": 0.9}, headers=ALICE).json()["models"]
    assert plain["ease"] == diverse["ease"]
    assert len(diverse["hybrid"]) == 5


def test_security_headers_and_timing(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Server-Timing"].startswith("app;dur=")


def test_trending_and_new_users_come_from_the_ml_forecast(client):
    """No popularity counts: the trending list and a brand-new user's first picks are what the hybrid
    predicts viewers will watch next (the crowd forecast), the latter for viewers who like the same genres."""
    top = client.get("/movies/trending", params={"k": 5})  # public: the landing page has no user
    assert top.status_code == 200 and len(top.json()) == 5
    assert {r["reason"] for r in top.json()} == {"Predicted next for many viewers"}

    who = {"X-User-Id": "newcomer"}
    first = client.post(
        "/onboarding", json={"favorite_genres": ["Comedy"], "ratings": []}, headers=who
    ).json()
    assert first and {r["reason"] for r in first} == {"Predicted for viewers who like Comedy"}
    picks = client.get("/recommendations", params={"k": 5}, headers=who).json()
    assert {r["reason"] for r in picks} == {"Predicted for viewers who like Comedy"}
    feed = client.get("/recommendations/feed", headers=who).json()["rows"]
    assert feed[-1]["key"] == "trending" and feed[-1]["title"] == "Predicted to watch next"


def test_why_this_pick_breaks_down_the_ranker_score(client):
    who = {"X-User-Id": "curious"}
    for m in (100, 101, 102, 103):
        client.put(f"/ratings/{m}", json={"rating": 5}, headers=who)
    top = client.get("/recommendations", params={"k": 3}, headers=who).json()[0]
    why = client.get(f"/recommendations/why/{top['movie_id']}", headers=who).json()
    assert why["available"] and why["rank"] >= 1 and why["candidates"] >= why["rank"]
    assert why["proposed_by"] and why["parts"]
    assert abs(sum(abs(p["share"]) for p in why["parts"]) - 1) < 1e-6  # shares of the whole score
    assert client.get("/recommendations/why/101", headers=who).json() == {  # rated: not a candidate
        "available": False,
        "rank": None,
        "candidates": None,
        "proposed_by": [],
        "parts": [],
    }


def test_watch_together_combines_every_members_fit(client):
    """Two friends with opposite tastes (films 0-9 vs 10-19 in the tiny catalogue): least misery finds the
    films that suit both best, most pleasure picks someone's favourites, and nobody's films come back."""
    a, b = [100, 101, 102, 103], [110, 111, 112, 113]
    body = {"include_me": False, "friends": [{"name": "Asha", "liked": a}, {"name": "Ravi", "liked": b}]}
    out = {}
    for strategy in ("average", "least_misery", "most_pleasure"):
        r = client.post("/group", json={**body, "strategy": strategy, "k": 5}, headers=ALICE)
        assert r.status_code == 200, r.text
        out[strategy] = r.json()
    assert out["average"]["members"] == ["Asha", "Ravi"]
    for res in out.values():
        assert not {i["movie"]["movie_id"] for i in res["items"]} & set(a + b)
        assert all(len(i["fits"]) == 2 and 0 <= min(i["fits"]) <= max(i["fits"]) <= 1 for i in res["items"])
    worst = lambda res: min(min(i["fits"]) for i in res["items"])  # noqa: E731
    assert worst(out["least_misery"]) >= worst(out["most_pleasure"])
    best = lambda res: max(max(i["fits"]) for i in res["items"])  # noqa: E731
    assert best(out["most_pleasure"]) == 1.0
    alone = client.post("/group", json={"include_me": True, "friends": []}, headers=ALICE)
    assert alone.status_code == 422


def test_new_films_row_is_empty_without_new_films(client):
    """The tiny bundle has no new films (`recsys newfilms` adds them): the endpoint answers, empty."""
    client.put("/ratings/100", json={"rating": 5}, headers=ALICE)
    out = client.get("/recommendations/new-films", headers=ALICE)
    assert out.status_code == 200
    assert out.json() == {"languages": [], "language": None, "refreshed_at": None, "items": []}
    unknown = client.get("/recommendations/new-films", params={"language": "Tamil"}, headers=ALICE)
    assert unknown.status_code == 422 and "available: none" in unknown.json()["detail"]
