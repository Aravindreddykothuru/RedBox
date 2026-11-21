"""Every metric checked against values worked out by hand (see comments)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from recsys.eval import metrics as M

# User A: targets at ranks 2 and 5, has 2 targets.
# User B: no target in the list, has 3 targets.
# User C: target at rank 1, has 1 target.
HITS = np.array(
    [
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ],
    dtype=bool,
)
N_REL = np.array([2, 3, 1])
L2_3 = math.log2(3)  # discount denominator at rank 2
L2_6 = math.log2(6)  # discount denominator at rank 5


def test_hit_rate():
    assert M.hit_rate(HITS, 3).tolist() == [1, 0, 1]
    assert M.hit_rate(HITS, 1).tolist() == [0, 0, 1]


def test_precision():
    np.testing.assert_allclose(M.precision(HITS, 3), [1 / 3, 0, 1 / 3])
    np.testing.assert_allclose(M.precision(HITS, 5), [2 / 5, 0, 1 / 5])


def test_recall():
    np.testing.assert_allclose(M.recall(HITS, N_REL, 3), [1 / 2, 0, 1])
    np.testing.assert_allclose(M.recall(HITS, N_REL, 5), [1, 0, 1])


def test_ndcg():
    # A@3: DCG = 1/log2(3); ideal = 2 targets at ranks 1,2 -> 1 + 1/log2(3)
    a3 = (1 / L2_3) / (1 + 1 / L2_3)
    # A@5: DCG = 1/log2(3) + 1/log2(6); same ideal
    a5 = (1 / L2_3 + 1 / L2_6) / (1 + 1 / L2_3)
    np.testing.assert_allclose(M.ndcg(HITS, N_REL, 3), [a3, 0, 1])
    np.testing.assert_allclose(M.ndcg(HITS, N_REL, 5), [a5, 0, 1])
    assert a3 == pytest.approx(0.386853, abs=1e-6)
    assert a5 == pytest.approx(0.624050, abs=1e-6)


def test_ndcg_ideal_is_capped_at_k():
    # 5 targets but k=2 and both top slots are hits -> perfect score
    hits = np.array([[1, 1, 0]], dtype=bool)
    assert M.ndcg(hits, np.array([5]), 2)[0] == pytest.approx(1.0)


def test_mrr():
    np.testing.assert_allclose(M.mrr(HITS, 5), [1 / 2, 0, 1])
    np.testing.assert_allclose(M.mrr(HITS, 1), [0, 0, 1])


def test_average_precision():
    # A@3: hit at rank 2 -> P@2 = 1/2, divided by min(2, 3) = 2 -> 0.25
    # A@5: hits at 2 and 5 -> (1/2 + 2/5) / 2 = 0.45
    np.testing.assert_allclose(M.average_precision(HITS, N_REL, 3), [0.25, 0, 1])
    np.testing.assert_allclose(M.average_precision(HITS, N_REL, 5), [0.45, 0, 1])


def test_k_out_of_range():
    with pytest.raises(ValueError):
        M.precision(HITS, 6)
    with pytest.raises(ValueError):
        M.precision(HITS, 0)


def test_gini_known_values():
    assert M.gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert M.gini(np.array([0, 0, 0, 10])) == pytest.approx(3 / 4)  # (n-1)/n when one has all
    assert M.gini(np.array([])) == 0.0


def test_coverage_and_exposure_gini():
    topk = np.array([[0, 1], [0, 1]])
    assert M.coverage(topk, n_items=4, k=2) == pytest.approx(2 / 4)
    # exposure counts [2, 2, 0, 0]: sorted [0,0,2,2] -> 2*(3*2+4*2)/(4*4) - 5/4 = 0.5
    assert M.exposure_gini(topk, n_items=4, k=2) == pytest.approx(0.5)
    assert M.coverage(np.array([[0, 1], [1, 2]]), n_items=4, k=2) == pytest.approx(3 / 4)


def test_novelty_and_popularity():
    item_pop = np.array([4, 2, 1, 1])  # out of 4 users
    topk = np.array([[0, 1], [1, 2]])
    # self-information: item0 -log2(1)=0, item1 -log2(1/2)=1, item2 -log2(1/4)=2
    np.testing.assert_allclose(M.novelty(topk, item_pop, n_users=4, k=2), [0.5, 1.5])
    np.testing.assert_allclose(M.average_popularity(topk, item_pop, k=2), [3.0, 1.5])


def test_intra_list_diversity():
    # genres: item0 {A}, item1 {A,B}, item2 {C}
    feats = sp.csr_matrix(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32))
    topk = np.array([[0, 1, 2], [0, 2, 1]])
    # k=2: [0,1] Jaccard 1/2 -> distance 0.5 ; [0,2] Jaccard 0 -> distance 1
    np.testing.assert_allclose(M.intra_list_diversity(topk, feats, 2), [0.5, 1.0])
    # k=3: pairs (0,1)=0.5, (0,2)=1, (1,2)=1 -> mean 2.5/3 for both orderings
    np.testing.assert_allclose(M.intra_list_diversity(topk, feats, 3), [2.5 / 3, 2.5 / 3])
