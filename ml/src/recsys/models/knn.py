"""Neighbourhood collaborative filtering (UserKNN and ItemKNN).

Similarity is cosine on the binary interaction matrix with a shrinkage term, which damps
similarities supported by only a few co-interactions:

    sim(a, b) = |a ∩ b| / (sqrt(|a|) * sqrt(|b|) + shrink)

Only the ``k`` most similar neighbours are kept; scores are similarity-weighted sums.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp

from recsys.data.split import TrainData
from recsys.models.base import Recommender, Validator


def _keep_top_k_per_column(sim: np.ndarray, k: int) -> sp.csr_matrix:
    """Sparse copy of ``sim`` keeping the k largest entries of every column."""
    n_rows, n_cols = sim.shape
    k = min(k, n_rows - 1)
    rows = np.argpartition(-sim, k - 1, axis=0)[:k, :]
    vals = np.take_along_axis(sim, rows, axis=0)
    cols = np.broadcast_to(np.arange(n_cols), rows.shape)
    out = sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=sim.shape)
    out.eliminate_zeros()
    return out


class ItemKNN(Recommender):
    """score(u, i) = sum of sim(j, i) over items j in u's history, keeping i's k nearest items j."""

    name = "itemknn"
    defaults: ClassVar[dict[str, Any]] = {"k": 100, "shrink": 10.0}

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        X = data.matrix.astype(np.float32).tocsr()
        co = (X.T @ X).toarray()  # co[i, j] = number of users who interacted with both i and j
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.params["shrink"] + 1e-12)
        np.fill_diagonal(sim, 0.0)  # an item is not its own neighbour
        self.similarity_ = _keep_top_k_per_column(sim.astype(np.float32), self.params["k"])
        self.X_ = X
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.X_[users] @ self.similarity_).toarray()


class UserKNN(Recommender):
    """score(u, i) = sum of sim(u, v) * x(v, i) over u's k most similar users v."""

    name = "userknn"
    defaults: ClassVar[dict[str, Any]] = {"k": 100, "shrink": 10.0}

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        self.X_ = data.matrix.astype(np.float32).tocsr()
        self.XT_ = self.X_.T.tocsr()
        self.norms_ = np.sqrt(np.asarray(self.X_.sum(axis=1)).ravel())  # binary: |row|^2 = row sum
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        n_users = self.X_.shape[0]
        # User-user similarities are computed per batch, so memory stays O(batch x users) at any scale.
        sim = (self.X_[users] @ self.XT_).toarray()
        sim /= self.norms_[users, None] * self.norms_[None, :] + self.params["shrink"] + 1e-12
        sim[np.arange(users.size), users] = 0.0  # a user is not their own neighbour
        k = min(self.params["k"], n_users - 1)
        nbrs = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        weights = np.take_along_axis(sim, nbrs, axis=1)
        rows = np.repeat(np.arange(users.size), k)
        W = sp.csr_matrix((weights.ravel(), (rows, nbrs.ravel())), shape=(users.size, n_users))
        return (W @ self.X_).toarray()
