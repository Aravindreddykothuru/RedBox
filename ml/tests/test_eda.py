from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.analysis.eda import activity_buckets, genre_table, gini, head_share, summarize
from recsys.data.movielens import load_ml1m


def test_gini_known_values():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    # one item holds everything: G = (n-1)/n
    assert gini(np.array([0, 0, 0, 10])) == pytest.approx(3 / 4)
    assert gini(np.array([])) == 0.0


def test_head_share():
    # 10 items; top 20% = 2 items holding 50+30 of 100 interactions
    counts = np.array([50, 30, 5, 5, 2, 2, 2, 2, 1, 1])
    assert head_share(counts, 0.2) == pytest.approx(0.8)


def test_activity_buckets_edges():
    counts = pd.Series([20, 21, 50, 51, 200, 201, 999])
    assert activity_buckets(counts) == {"1-20": 1, "21-50": 2, "51-200": 2, ">200": 2}


def test_summarize_on_fixture(ml1m_dir):
    s = summarize(load_ml1m(ml1m_dir), positive_threshold=4.0)
    assert s["n_users"] == 3
    assert s["n_items_rated"] == 3
    assert s["n_items_catalog"] == 4
    assert s["n_interactions"] == 6
    assert s["density"] == pytest.approx(6 / 9, abs=1e-5)
    assert s["n_positive"] == 3  # ratings 5, 4, 5
    # user 1 rated items 10 and 20 in the same second -> 2 of 6 interactions are tied
    assert s["tied_timestamp_share"] == pytest.approx(2 / 6, abs=1e-4)


def test_genre_table_counts_multi_genre_movies(ml1m_dir):
    table = genre_table(load_ml1m(ml1m_dir))
    # Drama: movies 20 and 30; ratings on 20 (x2) and 30 (x2)
    assert table.loc["Drama", "movies"] == 2
    assert table.loc["Drama", "ratings"] == 4
    assert table.loc["Documentary", "ratings"] == 0  # catalog-only genre still listed
