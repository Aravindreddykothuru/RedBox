"""BPR-MF: matrix factorization trained with Bayesian Personalized Ranking (Rendle et al., UAI 2009).

Each user u and item i get a ``dim``-dimensional vector; the score is their dot product. Training
samples triples (u, i, j) where u liked i and j is a random item u has not interacted with, and
pushes score(u, i) above score(u, j):

    loss = -log sigmoid(s_ui - s_uj) + reg/2 * (||p_u||^2 + ||q_i||^2 + ||q_j||^2)

The model is optimized for *ranking*, which is exactly what top-K recommendation needs.
Early stopping uses validation NDCG; the best epoch count is reused when refitting on train+val.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import torch
import torch.nn.functional as F

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator


def positive_keys(X) -> np.ndarray:
    """Sorted array of user * n_items + item for every interaction (fast membership tests)."""
    X = X.tocsr()
    users = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    return np.sort(users * X.shape[1] + X.indices.astype(np.int64))


def sample_negatives(
    users: np.ndarray, pos_keys: np.ndarray, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """One uniformly random item per user that the user has NOT interacted with (rejection sampling)."""
    neg = rng.integers(0, n_items, size=users.size)
    todo = np.arange(users.size)
    for _ in range(1000):
        keys = users[todo].astype(np.int64) * n_items + neg[todo]
        idx = np.minimum(np.searchsorted(pos_keys, keys), pos_keys.size - 1)
        clash = pos_keys[idx] == keys
        if not clash.any():
            return neg
        todo = todo[clash]
        neg[todo] = rng.integers(0, n_items, size=todo.size)
    raise RuntimeError("could not sample negatives; does a user interact with every item?")


class BPRMF(IterativeRecommender):
    name = "bpr"
    defaults: ClassVar[dict[str, Any]] = {
        "dim": 64,
        "lr": 1e-3,
        "reg": 1e-4,
        "batch_size": 4096,
        "epochs": 300,
        "eval_every": 5,
        "patience": 5,
        "seed": 42,
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        torch.manual_seed(p["seed"])
        self._rng = np.random.default_rng(p["seed"])
        X = data.matrix.tocsr()
        n_users, self._n_items = X.shape
        self._users = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(X.indptr))
        self._items = X.indices.astype(np.int64)
        self._pos_keys = positive_keys(X)

        self.user_emb_ = torch.nn.Parameter(torch.randn(n_users, p["dim"]) * 0.1)
        self.item_emb_ = torch.nn.Parameter(torch.randn(self._n_items, p["dim"]) * 0.1)
        self._optimizer = torch.optim.Adam([self.user_emb_, self.item_emb_], lr=p["lr"])
        self._train(validator)
        del self._optimizer, self._users, self._items, self._pos_keys  # training-only state
        return self

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        order = self._rng.permutation(self._users.size)
        total = 0.0
        for s in range(0, order.size, p["batch_size"]):
            b = order[s : s + p["batch_size"]]
            u, i = self._users[b], self._items[b]
            j = sample_negatives(u, self._pos_keys, self._n_items, self._rng)
            eu = self.user_emb_[torch.from_numpy(u)]
            ei = self.item_emb_[torch.from_numpy(i)]
            ej = self.item_emb_[torch.from_numpy(j)]
            margin = (eu * (ei - ej)).sum(dim=1)
            l2 = (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / (2 * b.size)
            loss = -F.logsigmoid(margin).mean() + p["reg"] * l2
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total += loss.item() * b.size
        return total / order.size

    def _snapshot(self):
        return self.user_emb_.detach().clone(), self.item_emb_.detach().clone()

    def _restore(self, state) -> None:
        with torch.no_grad():
            self.user_emb_.copy_(state[0])
            self.item_emb_.copy_(state[1])

    def score(self, users: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            u = self.user_emb_[torch.as_tensor(np.asarray(users), dtype=torch.long)]
            return (u @ self.item_emb_.T).numpy()
