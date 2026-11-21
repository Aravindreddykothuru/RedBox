"""Week 2 models: each test checks the math against an independent computation, or checks that the
model learns a structure planted in synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from recsys.data.split import TrainData
from recsys.models.ials import IALS, als_half_step
from recsys.models.lightgcn import LightGCN, SymmetricSpMM, normalized_adjacency, to_torch_sparse
from recsys.models.multvae import MultVAE
from recsys.models.neumf import NeuMF
from recsys.models.sasrec import SASRec, SASRecNet, left_padded


def train_data(dense: np.ndarray) -> TrainData:
    X = sp.csr_matrix(dense.astype(np.float32))
    u, i = X.nonzero()
    log = pd.DataFrame({"user_idx": u, "item_idx": i, "rating": 5.0, "timestamp": np.arange(u.size)})
    return TrainData(matrix=X, interactions=log)


def block_data(seed: int = 0, n_users: int = 40, n_items: int = 20):
    """Two taste groups (first half of users likes first half of items, and vice versa).
    Each user sees 7 of the 10 in-group items; the other 3 are held out."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_users, n_items))
    held_out = np.zeros_like(X, dtype=bool)
    half_items = n_items // 2
    for u in range(n_users):
        group = np.arange(half_items) if u < n_users // 2 else np.arange(half_items, n_items)
        seen = rng.choice(group, size=7, replace=False)
        X[u, seen] = 1
        held_out[u, np.setdiff1d(group, seen)] = True
    return X, held_out


def auc(scores: np.ndarray, X: np.ndarray, held_out: np.ndarray) -> float:
    """How often a held-out in-group item outranks an out-of-group item."""
    out_group = (X == 0) & ~held_out
    correct = total = 0
    for u in range(X.shape[0]):
        pos, neg = scores[u, held_out[u]], scores[u, out_group[u]]
        correct += (pos[:, None] > neg[None, :]).sum()
        total += pos.size * neg.size
    return correct / total


# ------------------------------------------------------------------------------------- iALS
def test_ials_half_step_equals_weighted_least_squares():
    """Each row solve must equal the textbook weighted ridge solution with an explicit confidence matrix,
    which checks the speed trick V^T C V = V^T V + alpha * V_u^T V_u."""
    rng = np.random.default_rng(0)
    R = sp.csr_matrix((rng.random((6, 9)) < 0.4).astype(float))
    Y = rng.normal(size=(9, 4))
    alpha, reg = 5.0, 0.3
    got = als_half_step(R, Y, alpha, reg)
    for r in range(6):
        p = R[r].toarray().ravel()
        C = np.diag(1 + alpha * p)
        expected = np.linalg.solve(Y.T @ C @ Y + reg * np.eye(4), Y.T @ C @ p) if p.any() else np.zeros(4)
        np.testing.assert_allclose(got[r], expected, rtol=1e-8, atol=1e-10)


def test_ials_objective_matches_brute_force():
    X, _ = block_data()
    model = IALS(dim=4, epochs=2, alpha=3.0, reg=0.5)
    model.fit(train_data(X))
    model._X = sp.csr_matrix(X)  # restore the training matrix (deleted after fit) for the check
    U, V = model.user_factors_, model.item_factors_
    S = U @ V.T
    brute = np.sum((1 + 3.0 * X) * (X - S) ** 2) + 0.5 * ((U**2).sum() + (V**2).sum())
    assert model.objective() == pytest.approx(brute, rel=1e-9)


def test_ials_learns_taste_groups():
    # Exact ALS solves overfit easily: with 8 factors and reg=0.1 the toy AUC drops to 0.93 because the
    # model memorizes individual missing cells. reg=1.0 (the default) gives ~0.99.
    X, held = block_data()
    model = IALS(dim=8, epochs=10, alpha=10.0, reg=1.0).fit(train_data(X))
    assert auc(model.score(np.arange(X.shape[0])), X, held) > 0.95


# --------------------------------------------------------------------------------- LightGCN
def test_normalized_adjacency_hand_computed():
    # user0-item0, user0-item1, user1-item0 -> degrees u0=2, u1=1, i0=2, i1=1
    A = normalized_adjacency(sp.csr_matrix(np.array([[1, 1], [1, 0]], dtype=np.float32))).toarray()
    # node order: u0, u1, i0, i1
    assert A[0, 2] == pytest.approx(1 / np.sqrt(2 * 2))
    assert A[0, 3] == pytest.approx(1 / np.sqrt(2 * 1))
    assert A[1, 2] == pytest.approx(1 / np.sqrt(1 * 2))
    assert A[0, 1] == 0 and A[2, 3] == 0  # no user-user or item-item edges
    np.testing.assert_allclose(A, A.T)


def test_symmetric_spmm_gradient_matches_autograd():
    """The fast custom backward must give exactly the gradient PyTorch computes with a dense matrix."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random((7, 5)) < 0.5).astype(np.float32))
    A = normalized_adjacency(X)
    x1 = torch.randn(12, 3, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    (SymmetricSpMM.apply(to_torch_sparse(A), x1) ** 2).sum().backward()
    (torch.from_numpy(A.toarray()) @ x2).pow(2).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad)


def test_lightgcn_zero_layers_is_plain_embeddings():
    X, _ = block_data()
    model = LightGCN(layers=0, epochs=1, batch_size=64).fit(train_data(X))
    torch.testing.assert_close(model.propagate(), model.emb_)


def test_lightgcn_learns_taste_groups():
    X, held = block_data()
    model = LightGCN(dim=16, layers=2, lr=0.05, epochs=30, batch_size=64).fit(train_data(X))
    assert auc(model.score(np.arange(X.shape[0])), X, held) > 0.95


# ------------------------------------------------------------------------------------ NeuMF
def test_neumf_learns_taste_groups_and_chunking_is_exact():
    X, held = block_data()
    model = NeuMF(gmf_dim=8, mlp_dims=[16, 8], lr=0.01, epochs=30, batch_size=128).fit(train_data(X))
    users = np.arange(X.shape[0])
    scores = model.score(users)
    assert auc(scores, X, held) > 0.9
    model.params["score_chunk"] = 7  # a different chunk size must not change any score
    np.testing.assert_allclose(model.score(users), scores, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------- Mult-VAE
def test_multvae_learns_taste_groups_and_scores_deterministically():
    X, held = block_data()
    model = MultVAE(hidden_dim=32, latent_dim=8, epochs=150, batch_size=20, lr=0.01, anneal_epochs=50)
    model.fit(train_data(X))
    users = np.arange(X.shape[0])
    scores = model.score(users)
    np.testing.assert_array_equal(model.score(users), scores)  # no sampling / dropout at prediction
    assert auc(scores, X, held) > 0.9


def test_multvae_kl_annealing_schedule():
    model = MultVAE(anneal_cap=0.2, anneal_epochs=10)
    model._steps_per_epoch = 5
    for step, beta in [(0, 0.0), (25, 0.1), (50, 0.2), (500, 0.2)]:
        model._step = step
        assert model._beta() == pytest.approx(beta)


# ----------------------------------------------------------------------------------- SASRec
def test_left_padding_keeps_most_recent_items():
    out = left_padded([np.array([1, 2, 3, 4]), np.array([7]), np.array([], dtype=int)], length=3)
    assert out.tolist() == [[2, 3, 4], [0, 0, 7], [0, 0, 0]]


def test_sasrec_is_causal():
    """The hidden state at position t must not depend on items after t (no peeking at the future)."""
    torch.manual_seed(0)
    net = SASRecNet(n_items=20, dim=16, max_len=6, blocks=2, heads=2, dropout=0.0).eval()
    a = torch.tensor([[0, 3, 5, 7, 9, 11]])
    b = torch.tensor([[0, 3, 5, 7, 2, 4]])  # same prefix, different future
    with torch.no_grad():
        ha, hb = net(a), net(b)
    torch.testing.assert_close(ha[:, :4], hb[:, :4])
    assert not torch.allclose(ha[:, 4:], hb[:, 4:])


def test_sasrec_is_padding_invariant():
    """A sequence must get the same hidden states however much padding its batch has. This is what
    makes it safe to trim each training batch to its longest sequence."""
    torch.manual_seed(0)
    net = SASRecNet(n_items=50, dim=16, max_len=10, blocks=2, heads=2, dropout=0.0).eval()
    short = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 3, 5, 7]])
    batch = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 3, 5, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        full = net(short)[0, -3:]
        torch.testing.assert_close(net(short[:, -3:])[0], full)  # trimmed padding
        torch.testing.assert_close(net(batch)[0, -3:], full)  # different batch-mates


def test_sasrec_initial_loss_is_sane():
    """With small initial weights, the BCE loss (positive + negative term) must start near 2*ln 2."""
    rows = [(u, (u + t) % 12, t) for u in range(30) for t in range(6)]
    log = pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"]).assign(rating=5.0)
    X = sp.csr_matrix((np.ones(len(log), np.float32), (log.user_idx, log.item_idx)), shape=(30, 12))
    model = SASRec(dim=16, blocks=1, max_len=6, epochs=1, batch_size=30, lr=1e-6)
    model.fit(TrainData(matrix=X, interactions=log))
    assert model.history_[0]["loss"] == pytest.approx(2 * np.log(2), abs=0.15)


def test_sasrec_shuffled_order_destroys_sequence_signal():
    """On data whose only signal is the order, shuffling the history must remove what SASRec can learn."""
    n_items, rng = 12, np.random.default_rng(0)
    rows = []
    for u in range(60):
        start = rng.integers(0, n_items)
        rows += [(u, (start + t) % n_items, t) for t in range(5)]
    log = pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"]).assign(rating=5.0)
    X = sp.csr_matrix((np.ones(len(log), np.float32), (log.user_idx, log.item_idx)), shape=(60, n_items))
    common = dict(dim=32, blocks=1, max_len=5, dropout=0.0, loss="ce", lr=0.01, epochs=80, batch_size=32)
    last = log.groupby("user_idx")["item_idx"].last().to_numpy()
    accuracy = {}
    for order in ("time", "shuffled"):
        model = SASRec(order=order, **common).fit(TrainData(matrix=X, interactions=log))
        predicted = model.recommend(np.arange(60), k=1, exclude=X)[:, 0]
        accuracy[order] = np.mean(predicted == (last + 1) % n_items)
    assert accuracy["time"] > 0.9
    assert accuracy["shuffled"] < 0.5


def test_markov_chain_counts_transitions():
    from recsys.models.markov import MarkovChain

    # user0: 0 -> 1 -> 2 ; user1: 0 -> 1 ; user2: 0 -> 2   => P(1|0) = 2/3, P(2|0) = 1/3, P(2|1) = 1
    log = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 2, 0, 1, 0, 2],
            "timestamp": range(7),
            "rating": 5.0,
        }
    )
    X = sp.csr_matrix((np.ones(7, np.float32), (log.user_idx, log.item_idx)), shape=(4, 3))
    model = MarkovChain(pop_weight=0.0).fit(TrainData(matrix=X, interactions=log))
    np.testing.assert_allclose(model.transitions_.toarray()[0], [0, 2 / 3, 1 / 3], rtol=1e-6)
    np.testing.assert_allclose(model.score(np.array([1]))[0], [0, 0, 1], rtol=1e-6)  # user1's last item is 1
    assert np.all(model.score(np.array([3]))[0] == 0)  # user without history (popularity weight 0)


@pytest.mark.parametrize("loss", ["ce", "bce"])
def test_sasrec_learns_successor_pattern(loss):
    """Every user watches consecutive items (k, k+1, k+2, ... mod 12); the next item must be predicted."""
    n_items, rng = 12, np.random.default_rng(0)
    rows = []
    for u in range(60):
        start = rng.integers(0, n_items)
        for t in range(5):
            rows.append((u, (start + t) % n_items, t))
    log = pd.DataFrame(rows, columns=["user_idx", "item_idx", "timestamp"]).assign(rating=5.0)
    X = sp.csr_matrix((np.ones(len(log), np.float32), (log.user_idx, log.item_idx)), shape=(60, n_items))
    model = SASRec(
        dim=32, blocks=1, heads=1, max_len=5, dropout=0.0, loss=loss, lr=0.01, epochs=80, batch_size=32
    )
    model.fit(TrainData(matrix=X, interactions=log))
    last = log.groupby("user_idx")["item_idx"].last().to_numpy()
    predicted = model.recommend(np.arange(60), k=1, exclude=X)[:, 0]
    accuracy = np.mean(predicted == (last + 1) % n_items)
    assert accuracy > 0.9
