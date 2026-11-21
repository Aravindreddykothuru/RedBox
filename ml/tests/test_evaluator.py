from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from recsys.eval.evaluator import evaluate, top_k
from recsys.models.base import Recommender


class FixedScores(Recommender):
    """Returns a fixed score matrix: lets us hand-compute the expected metrics."""

    name = "fixed"

    def __init__(self, scores: np.ndarray):
        super().__init__()
        self.scores = scores

    def fit(self, data, validator=None):
        return self

    def score(self, users):
        return self.scores[users]


SCORES = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # user 0: item 0 is history -> ranking 1,2,3,4,5
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # user 1: no targets -> not evaluated
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # user 2: ranking 5,4,3,2,1,0
    ],
    dtype=np.float32,
)
HISTORY = sp.csr_matrix(
    np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
)
# user 0 targets items 2 and 5 -> ranks 2 and 5 ; user 2 targets item 5 -> rank 1
TARGETS = sp.csr_matrix(
    np.array([[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=np.float32)
)


def test_top_k_sorted_best_first():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(s, 3).tolist() == [[1, 3, 2]]


def test_evaluate_matches_hand_computation():
    res = evaluate(FixedScores(SCORES), HISTORY, TARGETS, ks=(3, 5))
    m = res.metrics
    assert m["n_users"] == 2  # user 1 has no targets
    user0_ndcg3 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["ndcg@3"] == pytest.approx((user0_ndcg3 + 1.0) / 2)
    assert m["recall@5"] == pytest.approx(1.0)
    assert m["precision@5"] == pytest.approx((2 / 5 + 1 / 5) / 2)
    assert m["hit@3"] == pytest.approx(1.0)
    # top-3 lists: user0 [1,2,3], user2 [5,4,3] -> 5 distinct of 6 items
    assert m["coverage@3"] == pytest.approx(5 / 6)
    assert res.per_user["user_idx"].tolist() == [0, 2]


def test_history_items_are_never_recommended():
    # item 0 has the highest score for user 0 but is in the history
    res = evaluate(FixedScores(SCORES), HISTORY, TARGETS, ks=(5,))
    model = FixedScores(SCORES)
    recs = model.recommend(np.array([0]), 5, exclude=HISTORY)
    assert 0 not in recs[0]
    assert res.metrics["mrr@5"] == pytest.approx((1 / 2 + 1) / 2)


def test_evaluate_does_not_mutate_model_scores():
    scores = SCORES.copy()
    evaluate(FixedScores(scores), HISTORY, TARGETS, ks=(3,))
    np.testing.assert_array_equal(scores, SCORES)


def test_evaluate_rejects_bad_scores():
    bad = SCORES.copy()
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        evaluate(FixedScores(bad), HISTORY, TARGETS, ks=(3,))
    with pytest.raises(ValueError, match="shape"):
        evaluate(FixedScores(SCORES[:, :4]), HISTORY, TARGETS, ks=(3,))
