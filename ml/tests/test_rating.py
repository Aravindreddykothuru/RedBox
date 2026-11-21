from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.rating import BiasedMF, Biases, GlobalMean, mae, rmse


def test_rmse_mae_hand_computed():
    y, p = np.array([1, 2, 3, 4]), np.array([2, 2, 5, 4])
    assert rmse(y, p) == pytest.approx(np.sqrt((1 + 0 + 4 + 0) / 4))
    assert mae(y, p) == pytest.approx((1 + 0 + 2 + 0) / 4)


def _frame(u, i, r) -> pd.DataFrame:
    return pd.DataFrame({"user_idx": u, "item_idx": i, "rating": r, "timestamp": 0})


def test_biases_recover_additive_structure():
    """Ratings that are exactly mu + b_u + b_i (no noise, dense) must be predicted almost exactly."""
    rng = np.random.default_rng(0)
    bu, bi = rng.normal(0, 0.5, 30), rng.normal(0, 0.5, 20)
    u, i = np.meshgrid(np.arange(30), np.arange(20), indexing="ij")
    u, i = u.ravel(), i.ravel()
    r = 3.0 + bu[u] + bi[i]
    model = Biases(reg_user=0.0, reg_item=0.0, iterations=50).fit(_frame(u, i, r), 30, 20)
    assert rmse(np.clip(r, 1, 5), model.predict(u, i)) < 1e-6


def test_biased_mf_beats_biases_on_low_rank_data():
    rng = np.random.default_rng(1)
    P, Q = rng.normal(0, 0.8, (60, 3)), rng.normal(0, 0.8, (40, 3))
    u, i = np.meshgrid(np.arange(60), np.arange(40), indexing="ij")
    u, i = u.ravel(), i.ravel()
    r = np.clip(3.5 + (P[u] * Q[i]).sum(1) + rng.normal(0, 0.1, u.size), 1, 5)
    mask = rng.random(u.size) < 0.8
    train, test = _frame(u[mask], i[mask], r[mask]), (u[~mask], i[~mask], r[~mask])
    base = Biases().fit(train, 60, 40)
    mf = BiasedMF(dim=8, lr=0.02, reg=0.001, batch_size=256, epochs=60).fit(train, 60, 40)
    assert rmse(test[2], mf.predict(test[0], test[1])) < 0.6 * rmse(test[2], base.predict(test[0], test[1]))
    assert GlobalMean().fit(train, 60, 40).predict(test[0], test[1]).std() == 0.0
