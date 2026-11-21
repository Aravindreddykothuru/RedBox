"""First-order Markov chain: "what do people rate right after the item you rated last?"

    P(next = j | last = i) = count(i -> j) / count(i -> *)

counted over consecutive items in every user's chronological history. It is the simplest sequential
model, and the control experiment for SASRec. If this trivial model also beats the strong
non-sequential models by a wide margin, then the *order* of MovieLens ratings carries a strong signal
by itself. MovieLens ratings are entered in bulk sessions, so that order partly mirrors the order in
which the website displayed movies (Fan et al., 2023).

A tiny popularity term breaks ties among items never seen after the last item.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp

from recsys.data.split import TrainData
from recsys.models.base import Recommender, Validator


class MarkovChain(Recommender):
    name = "markov"
    defaults: ClassVar[dict[str, Any]] = {"pop_weight": 1e-6}

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        n_users, n_items = data.matrix.shape
        log = data.interactions  # chronological within each user
        u = log["user_idx"].to_numpy()
        i = log["item_idx"].to_numpy()
        same_user = u[1:] == u[:-1]
        src, dst = i[:-1][same_user], i[1:][same_user]
        counts = sp.csr_matrix((np.ones(src.size, np.float32), (src, dst)), shape=(n_items, n_items))
        row_sums = np.asarray(counts.sum(axis=1)).ravel()
        inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        self.transitions_ = sp.diags(inv.astype(np.float32)) @ counts  # row-normalized P(next | last)
        last = log.groupby("user_idx")["item_idx"].last()
        self.last_item_ = np.full(n_users, -1, dtype=np.int64)
        self.last_item_[last.index.to_numpy()] = last.to_numpy()
        pop = np.asarray(data.matrix.sum(axis=0)).ravel()
        self.popularity_ = (pop / max(pop.max(), 1)).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        last = self.last_item_[users]
        scores = np.tile(self.popularity_ * self.params["pop_weight"], (users.size, 1))
        has_last = last >= 0
        if has_last.any():
            scores[has_last] += self.transitions_[last[has_last]].toarray()
        return scores
