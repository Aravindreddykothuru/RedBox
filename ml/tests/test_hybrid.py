from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.experiments.runner import params_hash
from recsys.hybrid.features import Context, FeatureTable, build_features, last_items
from recsys.hybrid.ranker import HybridScores, mmr_order, score_matrix, train_ranker
from recsys.models.base import Recommender


class Fixed(Recommender):
    name = "fixed"

    def __init__(self, scores):
        super().__init__()
        self.scores = np.asarray(scores, dtype=np.float32)

    def fit(self, data, validator=None):
        return self

    def score(self, users):
        return self.scores[np.asarray(users)]


def _ctx(history: np.ndarray) -> Context:
    H = sp.csr_matrix(history.astype(np.float32))
    u, i = H.nonzero()
    log = pd.DataFrame({"user_idx": u, "item_idx": i})
    genres = sp.csr_matrix(np.eye(history.shape[1], 3, dtype=np.float32))  # item j has genre j (if j < 3)
    return Context(H, last_items(log, H.shape[0]), genres, np.arange(history.shape[1], dtype=float), None)


def test_candidates_union_excludes_history_and_labels_targets():
    history = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    a = Fixed([[9, 8, 7, 1, 1, 1], [1, 9, 8, 7, 1, 1]])  # top-2 user0: items 1,2 ; user1: 2,3
    b = Fixed([[1, 1, 1, 1, 9, 8], [8, 9, 1, 1, 1, 1]])  # top-2 user0: 4,5 ; user1: 0 (1 is history),2..
    targets = sp.csr_matrix(np.array([[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1]], dtype=np.float32))
    t = build_features({"a": a, "b": b}, np.array([0, 1]), _ctx(history), n_candidates=2, targets=targets)
    pairs = set(zip(t.users.tolist(), t.items.tolist(), strict=True))
    assert (0, 0) not in pairs and (1, 1) not in pairs  # history never becomes a candidate
    assert {i for u, i in pairs if u == 0} == {1, 2, 4, 5}
    assert t.X.shape == (len(t.users), len(t.names))
    labelled = {(u, i) for u, i, y in zip(t.users, t.items, t.labels, strict=True) if y == 1}
    assert labelled == {(0, 2), (0, 5)}
    # user1's target (item 5) is not among its candidates -> recall = 2 found / 3 targets
    assert t.candidate_recall == pytest.approx(2 / 3)


def test_ranker_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(300), 20)
    signal = rng.normal(size=users.size)
    y = (signal + rng.normal(0, 0.3, users.size) > 1.0).astype(np.float32)
    X = np.column_stack([signal, rng.normal(size=users.size)]).astype(np.float32)  # 2nd column is noise
    table = FeatureTable(users, np.tile(np.arange(20), 300), X, ["signal", "noise"], y, None)
    booster, info = train_ranker(table, rounds=200, early_stop=20, seed=0)
    pred = booster.predict(X)
    assert np.corrcoef(pred, signal)[0, 1] > 0.8
    gain = dict(zip(booster.feature_name(), booster.feature_importance("gain"), strict=True))
    assert gain["signal"] > 5 * gain["noise"]
    assert info["valid_users"] > 0 and len(info["held_users"]) == info["valid_users"]


def test_mmr_trades_relevance_for_diversity():
    genres = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)  # items 0 and 1 share a genre
    items, relevance = np.array([0, 1, 2]), np.array([1.0, 0.9, 0.1])
    assert mmr_order(items, relevance, genres, lam=1.0, k=3).tolist() == [0, 1, 2]
    # with diversity weighted in, the different-genre item 2 jumps ahead of the near-duplicate item 1
    assert mmr_order(items, relevance, genres, lam=0.5, k=3).tolist() == [0, 2, 1]


def test_score_matrix_and_hybrid_scores_round_trip():
    table = FeatureTable(
        np.array([0, 0, 1]), np.array([2, 5, 1]), np.zeros((3, 1), np.float32), ["x"], None, None
    )

    class Const:
        best_iteration = 1

        def predict(self, X, num_iteration=None):
            return np.array([0.2, 0.9, -0.5])

    S = score_matrix(table, Const(), (2, 6))
    dense = HybridScores(S).score(np.array([0, 1]))
    assert np.argmax(dense[0]) == 5 and np.argmax(dense[1]) == 1
    assert np.isneginf(dense[0, 0])  # non-candidates can never be recommended


def test_params_hash_is_order_independent_and_value_sensitive():
    assert params_hash({"a": 1, "b": [1, 2]}) == params_hash({"b": [1, 2], "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})
    assert params_hash({"lr": 0.001}) == params_hash({"lr": 1e-3})


def test_export_user_sampling_is_seeded_and_bounded():
    """Large datasets train the ranker and evaluate on a fixed random sample of users (bundle.py)."""
    import numpy as np

    from recsys.serving.bundle import _sample_users

    users = np.arange(0, 1000, 3)
    a = _sample_users(users, 50, np.random.default_rng(42))
    b = _sample_users(users, 50, np.random.default_rng(42))
    assert a.size == 50 and np.array_equal(a, b) and np.all(np.diff(a) > 0) and set(a) <= set(users)
    assert _sample_users(users, None, np.random.default_rng(0)) is users
    assert _sample_users(users, 10_000, np.random.default_rng(0)) is users
