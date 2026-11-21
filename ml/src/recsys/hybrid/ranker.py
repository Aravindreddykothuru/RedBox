"""Stage 2 (LightGBM LambdaRank ranker) and stage 3 (MMR diversity re-ranking)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

from recsys.hybrid.features import FeatureTable
from recsys.models.base import Recommender

DEFAULT_RANKER: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "verbose": -1,
}


def _groups(users: np.ndarray) -> np.ndarray:
    """LightGBM needs rows of one query (user) to be contiguous; returns the size of each group."""
    if users.size and np.any(np.diff(users) < 0):
        raise ValueError("feature rows must be sorted by user")
    _, counts = np.unique(users, return_counts=True)
    return counts


def train_ranker(
    table: FeatureTable,
    *,
    params: dict | None = None,
    rounds: int = 1000,
    early_stop: int = 50,
    seed: int = 42,
) -> tuple[lgb.Booster, dict]:
    """LambdaRank on users with at least one positive candidate; 15% of users held out for early stopping."""
    assert table.labels is not None
    users, y = table.users, table.labels
    has_pos = np.zeros(users.max() + 1, dtype=bool)
    has_pos[users[y > 0]] = True
    keep = has_pos[users]  # users without any positive candidate give LambdaRank no signal
    uniq = np.unique(users[keep])
    rng = np.random.default_rng(seed)
    held = rng.random(uniq.size) < 0.15
    held_users = np.zeros(users.max() + 1, dtype=bool)
    held_users[uniq[held]] = True
    tr = keep & ~held_users[users]
    va = keep & held_users[users]
    p = {**DEFAULT_RANKER, **(params or {}), "seed": seed}
    train_set = lgb.Dataset(table.X[tr], y[tr], group=_groups(users[tr]), feature_name=table.names)
    valid_set = lgb.Dataset(table.X[va], y[va], group=_groups(users[va]), reference=train_set)
    booster = lgb.train(
        p,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )
    info = {
        "best_iteration": booster.best_iteration,
        "valid_ndcg@10": booster.best_score["valid_0"]["ndcg@10"],
        "train_users": int(np.unique(users[tr]).size),
        "valid_users": int(held.sum()),
        "held_users": uniq[held],  # never seen by the ranker: used for the hybrid's validation score
    }
    return booster, info


def mmr_order(items: np.ndarray, relevance: np.ndarray, genres: np.ndarray, lam: float, k: int) -> np.ndarray:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998): greedily pick the item maximizing
    lam * relevance - (1 - lam) * (highest genre similarity to an item already picked)."""
    if lam >= 1.0 or items.size <= 1:
        return items[np.argsort(-relevance, kind="stable")][:k]
    rel = (relevance - relevance.min()) / (np.ptp(relevance) + 1e-9)  # comparable to similarity (0..1)
    g = genres[items]
    inter = g @ g.T
    size = g.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)  # Jaccard
    chosen: list[int] = []
    max_sim = np.zeros(items.size)
    available = np.ones(items.size, dtype=bool)
    for _ in range(min(k, items.size)):
        gain = np.where(available, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(gain))
        chosen.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return items[chosen]


class HybridScores(Recommender):
    """Wraps precomputed hybrid scores so the shared evaluator can score them like any model.
    Non-candidates get -inf: the hybrid only ever recommends from its candidate set."""

    name = "hybrid"

    def __init__(self, scores: sp.csr_matrix):
        super().__init__()
        self.scores_ = scores.tocsr()

    def fit(self, data, validator=None):
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        rows = self.scores_[np.asarray(users)]
        dense = np.full(rows.shape, -np.inf, dtype=np.float32)
        r = np.repeat(np.arange(rows.shape[0]), np.diff(rows.indptr))
        dense[r, rows.indices] = rows.data
        return dense


def score_matrix(
    table: FeatureTable,
    booster: lgb.Booster,
    shape: tuple[int, int],
    *,
    lam: float = 1.0,
    genres: np.ndarray | None = None,
    k: int = 20,
) -> sp.csr_matrix:
    """Ranker scores as a sparse (users x items) matrix; with lam < 1 the top-k of each user is
    re-ordered by MMR (positions k..n keep the ranker's order below them)."""
    pred = booster.predict(table.X, num_iteration=booster.best_iteration).astype(np.float32)
    if lam < 1.0:
        if genres is None:
            raise ValueError("MMR needs item genres")
        pred = pred.copy()
        bounds = np.flatnonzero(np.diff(table.users)) + 1
        for idx in np.split(np.arange(table.users.size), bounds):
            items = table.items[idx]
            picked = mmr_order(items, pred[idx], genres, lam, k)
            pos = {it: r for r, it in enumerate(picked)}
            top = pred[idx].max()
            new = pred[idx] - (np.ptp(pred[idx]) + 1.0)  # everything below the MMR list...
            for j, it in enumerate(items):
                if it in pos:
                    new[j] = top + (k - pos[it])  # ...and the MMR list on top, in MMR order
            pred[idx] = new
    # LightGBM scores can be negative: shift so that no candidate score is exactly 0 (sparse "missing")
    pred = pred - pred.min() + 1.0
    return sp.csr_matrix((pred, (table.users, table.items)), shape=shape)
