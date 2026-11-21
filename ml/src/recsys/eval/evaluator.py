"""Full-ranking evaluation, identical for every model.

For each user with held-out targets, the model scores *every* item, the user's known history is
removed, the rest is ranked, and the top K is compared with the targets. No negative sampling
(sampled metrics can change which model looks best; Krichene & Rendle, KDD 2020).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.eval import metrics as M

if TYPE_CHECKING:
    from recsys.models.base import Recommender


@dataclass
class EvalResult:
    metrics: dict[str, float]  # averaged over evaluated users (+ catalog-level coverage / gini)
    per_user: pd.DataFrame  # one row per evaluated user: for significance tests & bucket analysis
    seconds: float


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def _fill(dense: np.ndarray, rows: sp.csr_matrix, value) -> None:
    """dense[b, j] = value for every stored entry (b, j) of the CSR row block ``rows``."""
    r = np.repeat(np.arange(rows.shape[0]), np.diff(rows.indptr))
    dense[r, rows.indices] = value


def evaluate(
    model: Recommender,
    history: sp.csr_matrix,
    targets: sp.csr_matrix,
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    item_features: sp.spmatrix | None = None,
    batch_size: int = 1024,
) -> EvalResult:
    """Evaluate ``model`` on ``targets``, excluding each user's ``history`` from the ranking."""
    start = time.perf_counter()
    history, targets = history.tocsr(), targets.tocsr()
    if history.shape != targets.shape:
        raise ValueError(f"history {history.shape} and targets {targets.shape} differ in shape")
    n_users, n_items = history.shape
    n_rel_all = np.diff(targets.indptr)
    users = np.flatnonzero(n_rel_all > 0)
    if users.size == 0:
        raise ValueError("no user has any target to evaluate")

    k_max = max(ks)
    topk = np.empty((users.size, k_max), dtype=np.int64)
    hits = np.empty((users.size, k_max), dtype=bool)
    for s in range(0, users.size, batch_size):
        batch = users[s : s + batch_size]
        scores = np.array(model.score(batch), dtype=np.float32, copy=True)  # never mutate model state
        if scores.shape != (batch.size, n_items):
            raise ValueError(f"score() returned shape {scores.shape}, expected {(batch.size, n_items)}")
        if np.isnan(scores).any():
            raise ValueError(f"{type(model).__name__} produced NaN scores")
        _fill(scores, history[batch], -np.inf)  # already-known items are never recommended
        ranked = top_k(scores, k_max)
        relevant = np.zeros((batch.size, n_items), dtype=bool)
        _fill(relevant, targets[batch], True)
        topk[s : s + batch.size] = ranked
        hits[s : s + batch.size] = np.take_along_axis(relevant, ranked, axis=1)

    n_rel = n_rel_all[users]
    item_pop = np.asarray(history.sum(axis=0)).ravel()
    cols: dict[str, np.ndarray] = {
        "user_idx": users,
        "n_targets": n_rel,
        "n_history": np.diff(history.indptr)[users],
    }
    catalog: dict[str, float] = {}
    for k in ks:
        cols[f"ndcg@{k}"] = M.ndcg(hits, n_rel, k)
        cols[f"recall@{k}"] = M.recall(hits, n_rel, k)
        cols[f"hit@{k}"] = M.hit_rate(hits, k)
        cols[f"precision@{k}"] = M.precision(hits, k)
        cols[f"mrr@{k}"] = M.mrr(hits, k)
        cols[f"map@{k}"] = M.average_precision(hits, n_rel, k)
        cols[f"novelty@{k}"] = M.novelty(topk, item_pop, n_users, k)
        cols[f"arp@{k}"] = M.average_popularity(topk, item_pop, k)
        if item_features is not None:
            cols[f"ild@{k}"] = M.intra_list_diversity(topk, item_features, k)
        catalog[f"coverage@{k}"] = M.coverage(topk, n_items, k)
        catalog[f"gini@{k}"] = M.exposure_gini(topk, n_items, k)

    per_user = pd.DataFrame(cols)
    result = {c: float(per_user[c].mean()) for c in per_user.columns if "@" in c}
    result.update(catalog)
    result["n_users"] = float(users.size)
    return EvalResult(metrics=result, per_user=per_user, seconds=time.perf_counter() - start)
