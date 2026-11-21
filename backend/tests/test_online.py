"""Online evaluation: A/B-test assignment, impression logging, live metrics, and the schema migration."""

from __future__ import annotations

import math
import uuid

from sqlalchemy import create_engine, inspect, text

from app import experiments
from app.db import Database
from app.settings import Settings
from tests.conftest import make_client

ALICE = {"X-User-Id": "alice"}


def test_assignment_is_stable_and_balanced():
    settings = Settings(ab_experiment="exp-1", ab_variants={"control": 0.0, "diverse": 0.2})
    users = [uuid.uuid4() for _ in range(2000)]
    groups = [experiments.assign(u, settings) for u in users]
    assert groups == [experiments.assign(u, settings) for u in users]  # same group every time
    share = groups.count("diverse") / len(groups)
    assert 0.45 < share < 0.55
    other = Settings(ab_experiment="exp-2", ab_variants=settings.ab_variants)
    assert groups != [experiments.assign(u, other) for u in users]  # a new experiment reshuffles
    off = Settings(ab_experiment="", ab_variants=settings.ab_variants)
    assert {experiments.assign(u, off) for u in users[:50]} == {"control"}


def test_ctr_test_matches_a_hand_computed_z_test():
    r = experiments.ctr_test(100, 1000, 130, 1000)
    pooled = 230 / 2000
    z = 0.03 / math.sqrt(pooled * (1 - pooled) * 2 / 1000)
    assert math.isclose(r["z"], z) and math.isclose(r["difference"], 0.03)
    assert 0.03 < r["p_value"] < 0.04 and r["ci95"][0] < 0.03 < r["ci95"][1]
    assert experiments.ctr_test(0, 0, 1, 10)["p_value"] is None


def test_impressions_and_clicks_carry_the_group(client):
    for m in (100, 101, 102):
        client.put(f"/ratings/{m}", json={"rating": 5}, headers=ALICE)
    shown = client.get("/recommendations", params={"k": 5}, headers=ALICE).json()
    client.get("/recommendations/feed", headers=ALICE)
    client.post(
        "/events",
        json={"movie_id": shown[0]["movie_id"], "type": "click", "source_model": "hybrid", "position": 0},
        headers=ALICE,
    )
    client.get("/recommendations", params={"k": 5, "diversity": 0.5}, headers=ALICE)  # the user's own choice
    metrics = client.get("/ops/metrics").json()  # dev mode: everyone is an operator
    sources = metrics["by_source"]
    assert sources["hybrid"]["impressions"] == 10 and sources["hybrid"]["clicks"] == 1
    assert sources["trending"]["impressions"] > 0 and "top_picks" not in sources  # not double counted
    by_group = metrics["experiment"]["by_group"]
    assert by_group["manual"]["impressions"] == 5
    assigned = [g for g in by_group if g != "manual"]
    assert len(assigned) == 1 and by_group[assigned[0]]["clicks"] == 1
    assert metrics["catalogue"]["shown"] > 0


def test_live_metrics_are_for_operators_only(local_client):
    def register(email):
        r = local_client.post("/auth/register", json={"email": email, "password": "long enough pw"})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}, r.json()["user"]["is_admin"]

    user, user_is_admin = register("someone@example.com")
    ops, ops_is_admin = register("ops@example.com")
    assert not user_is_admin and ops_is_admin
    assert local_client.get("/ops/metrics", headers=user).status_code == 403
    assert local_client.get("/ops/metrics").status_code == 401
    assert local_client.get("/ops/metrics", headers=ops).json()["activity"]["accounts"] == 2


def test_existing_databases_get_new_columns(tmp_path):
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    old = create_engine(url)
    with old.begin() as conn:  # the events table as the first release created it, with one row
        conn.execute(
            text(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, user_id CHAR(32), movie_id INTEGER, "
                "type VARCHAR(20), source_model VARCHAR(32), position INTEGER, created_at DATETIME)"
            )
        )
        conn.execute(text("INSERT INTO events (user_id, movie_id, type) VALUES ('u', 1, 'click')"))
    Database(url).create_tables()
    assert "variant" in {c["name"] for c in inspect(create_engine(url)).get_columns("events")}
    with create_engine(url).connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM events")).scalar() == 1  # data kept


def test_ab_test_can_be_switched_off(bundle):
    with make_client(bundle, ab_experiment="") as c:
        c.put("/ratings/100", json={"rating": 5}, headers=ALICE)
        c.get("/recommendations", params={"k": 3}, headers=ALICE)
        assert set(c.get("/ops/metrics").json()["experiment"]["by_group"]) == {"control"}


def test_ratings_export_for_retraining_is_pseudonymous(local_client):
    def register(email):
        r = local_client.post("/auth/register", json={"email": email, "password": "long enough pw"})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    user, ops = register("viewer@example.com"), register("ops@example.com")
    local_client.put("/ratings/100", json={"rating": 4.5}, headers=user)
    local_client.put("/ratings/101", json={"rating": 2}, headers=user)
    assert local_client.get("/ops/ratings.csv", headers=user).status_code == 403
    r = local_client.get("/ops/ratings.csv", headers=ops)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0] == "user_key,movie_id,rating,timestamp" and len(lines) == 3
    rows = [line.split(",") for line in lines[1:]]
    assert {row[0] for row in rows} == {rows[0][0]} and len(rows[0][0]) == 16  # one stable pseudonym
    assert "viewer" not in r.text and "@" not in r.text  # no emails or ids leave the API
    assert sorted((int(m), float(v)) for _, m, v, _ in rows) == [(100, 4.5), (101, 2.0)]
