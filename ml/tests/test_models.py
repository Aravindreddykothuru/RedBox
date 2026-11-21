from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.data.split import TrainData
from recsys.models import build
from recsys.models.bpr import BPRMF, positive_keys, sample_negatives
from recsys.models.ease import EASE
from recsys.models.knn import ItemKNN, UserKNN
from recsys.models.popularity import MostPop


def train_data(dense: np.ndarray) -> TrainData:
    X = sp.csr_matrix(dense.astype(np.float32))
    u, i = X.nonzero()
    log = pd.DataFrame({"user_idx": u, "item_idx": i, "rating": 5.0, "timestamp": np.arange(u.size)})
    return TrainData(matrix=X, interactions=log)


# Users x items:      i0 i1 i2
TINY = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])  # u0, u1, u2
S01 = 2 / math.sqrt(6)  # cosine(i0, i1) = cosine(i0, i2) = cosine(u0, u2) = cosine(u1, u2)


def test_mostpop():
    model = MostPop().fit(train_data(TINY))
    np.testing.assert_allclose(model.score(np.array([0, 1])), [[3, 2, 2], [3, 2, 2]])


def test_itemknn_hand_computed():
    # cosine: sim(i0,i1) = sim(i0,i2) = 2/sqrt(6), sim(i1,i2) = 1/2
    full = ItemKNN(k=2, shrink=0).fit(train_data(TINY))
    np.testing.assert_allclose(full.score(np.array([0]))[0], [S01, S01, S01 + 0.5], rtol=1e-5)
    # k=1: item 2 keeps only its nearest neighbour (item 0), so sim(i1, i2) is dropped
    top1 = ItemKNN(k=1, shrink=0).fit(train_data(TINY))
    assert top1.score(np.array([0]))[0][2] == pytest.approx(S01, rel=1e-5)


def test_userknn_hand_computed():
    # cosine: sim(u0,u1) = 1/2, sim(u0,u2) = 2/sqrt(6); a user is never their own neighbour
    full = UserKNN(k=2, shrink=0).fit(train_data(TINY))
    np.testing.assert_allclose(full.score(np.array([0]))[0], [0.5 + S01, S01, 0.5 + S01], rtol=1e-5)
    top1 = UserKNN(k=1, shrink=0).fit(train_data(TINY))
    np.testing.assert_allclose(top1.score(np.array([0]))[0], [S01, S01, S01], rtol=1e-5)


def test_ease_equals_per_item_ridge_regression():
    """EASE's closed form must equal solving, for every item j, a ridge regression of
    column j on all other columns (that is exactly what the diag(B)=0 constraint means)."""
    rng = np.random.default_rng(0)
    X = (rng.random((40, 8)) < 0.35).astype(np.float64)
    lam = 3.0
    B = EASE(l2=lam).fit(train_data(X)).B_.astype(np.float64)
    n_items = X.shape[1]
    for j in range(n_items):
        others = [c for c in range(n_items) if c != j]
        A = X[:, others]
        expected = np.linalg.solve(A.T @ A + lam * np.eye(n_items - 1), A.T @ X[:, j])
        np.testing.assert_allclose(B[others, j], expected, rtol=1e-4, atol=1e-6)
        assert B[j, j] == 0.0


def test_negative_sampler_never_returns_positives():
    rng = np.random.default_rng(1)
    X = sp.csr_matrix((rng.random((50, 30)) < 0.6).astype(np.float32))
    keys = positive_keys(X)
    users = rng.integers(0, 50, size=20_000)
    neg = sample_negatives(users, keys, 30, rng)
    assert not X[users, neg].any()


def _block_data(seed: int = 0):
    """Two taste groups: users 0-19 like items 0-9, users 20-39 like items 10-19.
    Each user sees 7 of their 10 in-group items; the other 3 are held out."""
    rng = np.random.default_rng(seed)
    X = np.zeros((40, 20))
    held_out = np.zeros_like(X, dtype=bool)
    for u in range(40):
        group = np.arange(10) if u < 20 else np.arange(10, 20)
        seen = rng.choice(group, size=7, replace=False)
        X[u, seen] = 1
        held_out[u, np.setdiff1d(group, seen)] = True
    return X, held_out


def test_bpr_learns_taste_groups():
    X, held_out = _block_data()
    model = BPRMF(dim=8, lr=0.05, reg=1e-4, batch_size=64, epochs=40, seed=0).fit(train_data(X))
    scores = model.score(np.arange(40))
    out_group = (X == 0) & ~held_out
    # AUC: held-out in-group items should outrank out-of-group items
    correct = total = 0
    for u in range(40):
        pos, neg = scores[u, held_out[u]], scores[u, out_group[u]]
        correct += (pos[:, None] > neg[None, :]).sum()
        total += pos.size * neg.size
    assert correct / total > 0.95


def test_bpr_early_stopping_records_best_epoch():
    X, _ = _block_data()
    calls = iter([0.1, 0.3, 0.2, 0.2, 0.2])  # best at the 2nd evaluation, then no improvement
    model = BPRMF(dim=4, epochs=50, eval_every=2, patience=3, batch_size=64, seed=0)
    model.fit(train_data(X), validator=lambda m: next(calls))
    assert model.best_epoch_ == 4
    assert model.refit_params() == {"epochs": 4}
    assert len(model.history_) == 10  # stopped after 3 evaluations without improvement


def test_recommend_excludes_history():
    data = train_data(TINY)
    model = MostPop().fit(data)
    recs = model.recommend(np.arange(3), k=1, exclude=data.matrix)
    for u, (item,) in enumerate(recs):
        assert TINY[u, item] == 0 or TINY[u].all()


def test_registry():
    assert isinstance(build("ease", l2=10), EASE)
    with pytest.raises(TypeError, match="unknown parameters"):
        build("ease", lambda_=10)
    with pytest.raises(KeyError, match="Unknown model"):
        build("does-not-exist")
