"""The torch-free serving scorers must reproduce the trained models exactly, and the engine must never
recommend something the user already rated."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from recsys.data.split import TrainData
from recsys.models.ease import EASE
from recsys.models.ials import IALS, als_half_step
from recsys.models.markov import MarkovChain
from recsys.models.sasrec import SASRecNet
from recsys.serving.bundle import sequences_of
from recsys.serving.engine import Engine
from recsys.serving.scorers import EASEScorer, IALSScorer, MarkovScorer, SASRecScorer
from recsys.serving.testing import block_matrix, make_tiny_bundle


def _data(X: np.ndarray) -> TrainData:
    M = sp.csr_matrix(X.astype(np.float32))
    u, i = M.nonzero()
    log = pd.DataFrame({"user_idx": u, "item_idx": i, "rating": 5.0, "timestamp": np.arange(u.size)})
    return TrainData(matrix=M, interactions=log)


def test_numpy_sasrec_matches_pytorch():
    torch.manual_seed(0)
    cfg = {"dim": 16, "heads": 2, "blocks": 2, "max_len": 8}
    net = SASRecNet(n_items=30, dim=16, max_len=8, blocks=2, heads=2, dropout=0.1).eval()
    scorer = SASRecScorer.from_torch(net, cfg)
    seq = np.array([[0, 0, 0, 3, 5, 7, 9, 11], [1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 4]])
    with torch.no_grad():
        expected = net(torch.from_numpy(seq)).numpy()
    np.testing.assert_allclose(scorer.hidden(seq), expected, rtol=1e-4, atol=1e-5)
    # trimmed padding (as used for live requests) gives the same last state
    np.testing.assert_allclose(scorer.hidden(seq[:1, -5:])[:, -1], expected[:1, -1], rtol=1e-4, atol=1e-5)


def test_ease_and_markov_scorers_match_models():
    X, _ = block_matrix()
    data = _data(X)
    seqs = sequences_of(data.interactions, X.shape[0])
    users = np.arange(X.shape[0])
    ease = EASE(l2=5.0).fit(data)
    np.testing.assert_allclose(
        EASEScorer(ease.B_).set_history(data.matrix, seqs).score(users), ease.score(users), rtol=1e-5
    )
    mc = MarkovChain().fit(data)
    scorer = MarkovScorer(mc.transitions_, mc.popularity_, mc.params["pop_weight"]).set_history(
        data.matrix, seqs
    )
    np.testing.assert_allclose(scorer.score(users), mc.score(users), rtol=1e-6)


def test_ials_fold_in_is_the_exact_als_user_step():
    X, _ = block_matrix()
    data = _data(X)
    model = IALS(dim=4, epochs=3, alpha=5.0, reg=0.5).fit(data)
    scorer = IALSScorer(model.item_factors_, 5.0, 0.5).set_history(data.matrix, [])
    expected = als_half_step(data.matrix.astype(np.float64), model.item_factors_, 5.0, 0.5)
    np.testing.assert_allclose(scorer.user_vectors(np.arange(X.shape[0])), expected, rtol=1e-8, atol=1e-10)


def test_engine_end_to_end(tmp_path):
    engine = Engine(make_tiny_bundle(tmp_path / "b"))
    history, disliked = [0, 1, 2, 3], {4}
    recs = engine.recommend(history, exclude=disliked, k=5)
    assert 0 < len(recs) <= 5
    assert not {r.item_idx for r in recs} & (set(history) | disliked)  # never re-recommend rated items
    assert all(r.reason for r in recs)
    assert all(r.item_idx < 10 for r in recs[:3])  # this user is in taste group 1 (items 0..9)
    diverse = engine.recommend(history, exclude=disliked, k=5, lam=0.5)
    assert not {r.item_idx for r in diverse} & (set(history) | disliked)
    for model in engine.models:
        assert engine.recommend(history, k=3, model=model)
    cold = engine.recommend([], k=3)
    assert [r.reason for r in cold] == ["Popular with viewers right now"] * 3
    assert len(engine.similar(0, k=4)) == 4 and len(engine.similar(0, k=4, mode="content")) == 4
    anchor, row = engine.because_you_liked(history, k=3)
    assert anchor.item_idx == 3 and all(r.item_idx not in history for r in row)
    assert set(engine.lab_compare(history, k=2)) == {"hybrid", "ease", "markov", "mostpop"}
    assert engine.to_idx([100, 105, 999]) == [0, 5]
    assert engine.dataset_history(1) and len(engine.dataset_user_ids()) == 40


def test_live_features_equal_offline_features(tmp_path):
    """No train/serve skew: features for one user computed live must equal the batch features the ranker
    was trained and evaluated with (popularity and similarity are global, not from one user)."""
    from recsys.hybrid.features import Context, build_features
    from recsys.serving.engine import dataset_matrix

    engine = Engine(make_tiny_bundle(tmp_path / "b"))
    M = dataset_matrix(engine._dataset_users, engine.n_items)
    seqs = [np.asarray(engine.dataset_history(u)) for u in engine.dataset_user_ids()]
    for s in engine.scorers.values():
        s.set_history(M, seqs)
    last = np.array([s[-1] for s in seqs])
    ctx = Context(M, last, engine.genres, engine.years, None)  # offline: statistics from all users
    batch = build_features(engine.scorers, np.arange(M.shape[0]), ctx, n_candidates=engine.n_candidates)
    for u in (0, 7, 25):
        online = engine.features(list(seqs[u]))
        rows = batch.users == u
        order_b, order_o = np.argsort(batch.items[rows]), np.argsort(online.items)
        np.testing.assert_array_equal(batch.items[rows][order_b], online.items[order_o])
        np.testing.assert_allclose(batch.X[rows][order_b], online.X[order_o], rtol=1e-5, atol=1e-6)


def test_serving_path_never_imports_pytorch(tmp_path):
    """The deployed API must run without PyTorch installed (small image, fast start)."""
    bundle = make_tiny_bundle(tmp_path / "b")
    code = (
        "import sys; from recsys.serving.engine import Engine; "
        f"e = Engine(r'{bundle}'); e.recommend([0, 1, 2], k=3); "
        "assert 'torch' not in sys.modules, 'serving imported torch'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_bundle_without_histories_serves_identical_recommendations(tmp_path):
    """Public deployments drop the raw MovieLens histories (dataset_users.npz) and ship only aggregate
    item statistics; recommendations must not change."""
    import shutil

    from recsys.serving.engine import Engine, write_item_stats
    from recsys.serving.testing import make_tiny_bundle

    full = make_tiny_bundle(tmp_path / "full")
    write_item_stats(full)
    public = tmp_path / "public"
    shutil.copytree(full, public)
    (public / "dataset_users.npz").unlink()
    a, b = Engine(full), Engine(public)
    for history in ([0, 1, 2], [12, 15], [3]):
        assert [r.item_idx for r in a.recommend(history, k=8)] == [
            r.item_idx for r in b.recommend(history, k=8)
        ]
    assert b.dataset_user_ids() == [] and a.dataset_user_ids()
