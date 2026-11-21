"""EASE^R: Embarrassingly Shallow AutoEncoder (Steck, WWW 2019).

Learns an item-item weight matrix B that reconstructs each user's interactions from their other
interactions:

    min_B ||X - X B||^2 + l2 * ||B||^2   subject to   diag(B) = 0

The constraint stops the trivial answer B = I ("predict an item from itself"). The problem has a
closed-form solution, so training is a single matrix inversion:

    P = (X^T X + l2 * I)^-1,    B_ij = -P_ij / P_jj  (i != j),   B_jj = 0

Scoring a user only needs their interaction vector (x_u @ B), so brand-new users can be scored
immediately without retraining, which the live app relies on.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np

from recsys.data.split import TrainData
from recsys.models.base import Recommender, Validator


class EASE(Recommender):
    name = "ease"
    defaults: ClassVar[dict[str, Any]] = {"l2": 500.0}

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        X = data.matrix
        gram = (X.T @ X).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.params["l2"]
        P = np.linalg.inv(gram)
        B = P / (-np.diag(P))  # divides column j by -P_jj
        B[np.diag_indices_from(B)] = 0.0
        self.B_ = B.astype(np.float32)
        self.X_ = X.tocsr()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.B_)
