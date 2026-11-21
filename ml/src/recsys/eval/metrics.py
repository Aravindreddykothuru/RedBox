"""Top-K recommendation metrics (binary relevance), one value per user.

Accuracy metrics take
    hits  (n_users, K_max) bool: hits[u, r] is True if the item at rank r+1 is a held-out target of u
    n_rel (n_users,) int:        number of held-out targets of u (must be >= 1)
and return a per-user array, so results can be averaged, bucketed (cold-start analysis) or used in
paired significance tests.

Beyond-accuracy metrics take
    topk  (n_users, K_max) int:  recommended item indices, best first
and describe *what* gets recommended (coverage, novelty, popularity bias, diversity).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _top(hits: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= hits.shape[1]:
        raise ValueError(f"k={k} must be between 1 and {hits.shape[1]}")
    return hits[:, :k]


# --------------------------------------------------------------------------- accuracy
def hit_rate(hits: np.ndarray, k: int) -> np.ndarray:
    """1 if at least one target is in the top k, else 0."""
    return _top(hits, k).any(axis=1).astype(np.float64)


def precision(hits: np.ndarray, k: int) -> np.ndarray:
    """Share of the k recommendations that are targets."""
    return _top(hits, k).sum(axis=1) / k


def recall(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Share of the user's targets that appear in the top k."""
    return _top(hits, k).sum(axis=1) / n_rel


def ndcg(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Normalized discounted cumulative gain: hits near the top count more (1 / log2(rank + 1)).

    Normalized by the best possible DCG, i.e. min(n_rel, k) targets placed at the very top.
    """
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (_top(hits, k) * discount).sum(axis=1)
    idcg = np.cumsum(discount)[np.minimum(n_rel, k) - 1]
    return dcg / idcg


def mrr(hits: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank of the first target within the top k (0 if none)."""
    top = _top(hits, k)
    first = top.argmax(axis=1)
    return np.where(top.any(axis=1), 1.0 / (first + 1), 0.0)


def average_precision(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """AP@k: mean of precision@i over the ranks i that hold a target, divided by min(n_rel, k)."""
    top = _top(hits, k).astype(np.float64)
    precision_at_i = np.cumsum(top, axis=1) / np.arange(1, k + 1)
    return (precision_at_i * top).sum(axis=1) / np.minimum(n_rel, k)


# --------------------------------------------------------------------- beyond accuracy
def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array: 0 = perfectly equal, -> 1 = one entry has everything."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * x)) / (n * x.sum()) - (n + 1) / n)


def coverage(topk: np.ndarray, n_items: int, k: int) -> float:
    """Share of the catalog recommended to at least one user."""
    return np.unique(topk[:, :k]).size / n_items


def exposure_gini(topk: np.ndarray, n_items: int, k: int) -> float:
    """Gini of how often each catalog item is recommended (higher = attention concentrated on few items)."""
    return gini(np.bincount(topk[:, :k].ravel(), minlength=n_items))


def novelty(topk: np.ndarray, item_pop: np.ndarray, n_users: int, k: int) -> np.ndarray:
    """Mean self-information -log2(p_i) of recommended items, p_i = share of users who know item i."""
    info = -np.log2(np.maximum(item_pop, 1) / n_users)
    return info[topk[:, :k]].mean(axis=1)


def average_popularity(topk: np.ndarray, item_pop: np.ndarray, k: int) -> np.ndarray:
    """Average number of training interactions of the recommended items (popularity bias)."""
    return item_pop[topk[:, :k]].mean(axis=1).astype(np.float64)


def intra_list_diversity(topk: np.ndarray, item_features: sp.spmatrix | np.ndarray, k: int) -> np.ndarray:
    """Mean pairwise Jaccard distance between the feature sets (e.g. genres) of the k recommended items."""
    if k < 2:
        return np.zeros(topk.shape[0])
    feats = item_features.toarray() if sp.issparse(item_features) else np.asarray(item_features)
    f = feats.astype(np.float32)[topk[:, :k]]  # (users, k, features)
    inter = np.einsum("ukf,ulf->ukl", f, f)
    size = f.sum(axis=2)
    union = size[:, :, None] + size[:, None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    off_diagonal = ~np.eye(k, dtype=bool)
    return (1.0 - sim)[:, off_diagonal].mean(axis=1).astype(np.float64)
