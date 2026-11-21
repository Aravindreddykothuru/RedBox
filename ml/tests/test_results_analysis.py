from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.analysis.results import add_activity_bucket, by_activity, paired_tests


def _per_user(values: dict[str, np.ndarray], n_history: np.ndarray) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"user_idx": np.arange(len(v)), "n_history": n_history, "ndcg@10": v, "model": m})
        for m, v in values.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_activity_bucket_edges():
    df = pd.DataFrame({"n_history": [1, 20, 21, 50, 51, 200, 201, 5000]})
    assert add_activity_bucket(df)["bucket"].astype(str).tolist() == [
        "1-20", "1-20", "21-50", "21-50", "51-200", "51-200", ">200", ">200",
    ]  # fmt: skip


def test_by_activity_means_and_user_counts():
    hist = np.array([10, 10, 100, 300])
    df = _per_user({"a": np.array([0.1, 0.3, 0.5, 0.7])}, hist)
    table = by_activity(df)
    assert table.loc["1-20", "a"] == pytest.approx(0.2)
    assert table.loc[">200", "a"] == pytest.approx(0.7)
    assert table["users"].tolist() == [2, 0, 1, 1]


def test_paired_tests_detects_consistent_difference_and_holm_is_conservative():
    rng = np.random.default_rng(0)
    base = rng.random(60)
    df = _per_user(
        {"ref": base + 0.1, "worse": base, "same": base + rng.normal(0, 0.001, 60)}, np.full(60, 30)
    )
    out = paired_tests(df, reference="ref").set_index("comparison")
    assert out.loc["ref vs worse", "significant (p<0.05)"]
    assert out.loc["ref vs worse", "mean difference"] == pytest.approx(0.1)
    assert (out["p-value (Holm)"] >= out["p-value"]).all()
    assert (out["p-value (Holm)"] <= 1).all()
