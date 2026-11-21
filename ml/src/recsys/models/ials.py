"""iALS: implicit-feedback matrix factorization by alternating least squares (Hu, Koren, Volinsky, ICDM 2008).

Every (user, item) cell counts, not just the observed ones: a liked item is a positive with high
confidence c = 1 + alpha, an unobserved item a weak negative with confidence 1.

    min_{U,V}  sum_{u,i} c_ui (p_ui - u_u . v_i)^2 + reg (||U||^2 + ||V||^2),   p_ui = 1 if liked else 0

With V fixed, every user vector has a closed-form solution, and vice versa, so training alternates
between the two exact solves. The trick that makes this fast: V^T C_u V = V^T V + alpha * V_u^T V_u,
where V_u holds only the items user u liked, so each solve touches only that user's own items.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator


def als_half_step(R: sp.csr_matrix, Y: np.ndarray, alpha: float, reg: float) -> np.ndarray:
    """Solve every row factor given the column factors Y (one half of an ALS iteration)."""
    d = Y.shape[1]
    gram = Y.T @ Y + reg * np.eye(d)
    out = np.zeros((R.shape[0], d))
    for r in range(R.shape[0]):
        idx = R.indices[R.indptr[r] : R.indptr[r + 1]]
        if idx.size == 0:
            continue
        Yr = Y[idx]
        A = gram + alpha * (Yr.T @ Yr)
        b = (1.0 + alpha) * Yr.sum(axis=0)
        out[r] = np.linalg.solve(A, b)
    return out


class IALS(IterativeRecommender):
    name = "ials"
    defaults: ClassVar[dict[str, Any]] = {
        "dim": 64,
        "alpha": 10.0,
        "reg": 1.0,
        "epochs": 30,
        "eval_every": 1,
        "patience": 3,
        "seed": 42,
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        rng = np.random.default_rng(p["seed"])
        self._X = data.matrix.tocsr().astype(np.float64)
        self._XT = self._X.T.tocsr()
        n_users, n_items = self._X.shape
        self.user_factors_ = rng.normal(0, 0.01, (n_users, p["dim"]))
        self.item_factors_ = rng.normal(0, 0.01, (n_items, p["dim"]))
        self._train(validator)
        del self._X, self._XT
        return self

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        self.user_factors_ = als_half_step(self._X, self.item_factors_, p["alpha"], p["reg"])
        self.item_factors_ = als_half_step(self._XT, self.user_factors_, p["alpha"], p["reg"])
        return self.objective() / self._X.nnz

    def objective(self) -> float:
        """Training objective without materializing the dense user x item matrix:
        sum over all cells of s^2 equals trace((U^T U)(V^T V)); observed cells add (1+a)(1-s)^2 - s^2."""
        U, V, a = self.user_factors_, self.item_factors_, self.params["alpha"]
        all_cells = float(np.sum((U.T @ U) * (V.T @ V)))
        rows = np.repeat(np.arange(self._X.shape[0]), np.diff(self._X.indptr))
        s = np.einsum("ij,ij->i", U[rows], V[self._X.indices])
        observed = float(np.sum((1 + a) * (1 - s) ** 2 - s**2))
        return all_cells + observed + self.params["reg"] * float((U**2).sum() + (V**2).sum())

    def _snapshot(self):
        return self.user_factors_.copy(), self.item_factors_.copy()

    def _restore(self, state) -> None:
        self.user_factors_, self.item_factors_ = state

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.user_factors_[np.asarray(users)] @ self.item_factors_.T).astype(np.float32)
