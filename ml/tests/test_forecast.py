"""The crowd forecast: trending and brand-new users from what the hybrid predicts viewers watch next."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.serving.engine import Engine
from recsys.serving.forecast import add_forecast_to_bundle, evaluate_forecast
from recsys.serving.testing import make_tiny_bundle


def test_forecast_replaces_popularity_for_trending_and_new_users(tmp_path):
    path = make_tiny_bundle(tmp_path / "tiny")
    assert Engine(path).crowd is None  # without a forecast: popularity, as before

    fc, others = add_forecast_to_bundle(path, n_viewers=20, seed=0)
    assert fc["crowd"].shape == (20,) and fc["by_genre"].shape == (3, 20)
    assert int(fc["n_viewers"]) == 20 and len(others) == 20  # the other half of the 40 viewers
    assert 0 < fc["crowd"].sum() <= 20 * (1 / np.log2(np.arange(2, 12))).sum() + 1e-4  # at most 20 top-10s

    e = Engine(path)
    top = e.trending(k=5)
    picked = [r.item_idx for r in top]
    rest = np.delete(e.crowd, picked)
    assert min(e.crowd[picked]) >= rest.max() - 1e-6  # the films most often predicted next
    assert {r.reason for r in top} == {"Predicted next for many viewers"}
    assert {r.item_idx for r in e.trending(k=5, exclude={picked[0]})}.isdisjoint({picked[0]})

    new = e.recommend([], k=5, genres=["Comedy", "Drama"])
    assert {r.reason for r in new} == {"Predicted for viewers who like Comedy and Drama"}
    many = e.recommend([], k=3, genres=["Action", "Comedy", "Drama"])
    assert many[0].reason == "Predicted for viewers who like Action, Comedy and more"
    assert e.recommend([], k=3, genres=["Not a genre"])[0].reason == "Predicted next for many viewers"
    assert e.manifest["forecast"]["viewers"] == 20


def test_forecast_evaluation_reports_both_uses(tmp_path):
    path = make_tiny_bundle(tmp_path / "tiny")
    fc, others = add_forecast_to_bundle(path, n_viewers=20, seed=0)
    e = Engine(path)
    rng = np.random.default_rng(0)
    targets = sp.csr_matrix((rng.random((40, 20)) < 0.15).astype(np.float32))  # "liked next"
    table = evaluate_forecast(e, fc, targets, others)
    assert set(table["use"]) == {"trending", "cold start"}
    assert set(table.loc[table["use"] == "cold start", "method"]) == {
        "popularity",
        "popular in the chosen genres",
        "ML crowd forecast (no genres)",
        "ML viewers like you (chosen genres)",
    }
    scores = table[["NDCG@10"]].to_numpy()
    assert np.all((scores >= 0) & (scores <= 1))
