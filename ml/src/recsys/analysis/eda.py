"""Dataset statistics that drive modelling decisions.

Each number here answers a design question for Week 1+:
- sparsity / long tail / Gini   -> why plain CF struggles, why we report coverage & novelty
- positives per user           -> can every user be split 80/10/10 at the >=4 threshold?
- activity buckets             -> groups for the cold-start analysis (experiment E5)
- timestamp ties / sessions    -> how to break ties in the chronological split, and
                                  whether sequence order (SASRec) is meaningful
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.schema import RecDataset
from recsys.eval.metrics import gini

__all__ = ["ACTIVITY_BUCKETS", "activity_buckets", "genre_table", "gini", "head_share", "summarize"]

ACTIVITY_BUCKETS = [(1, 20), (21, 50), (51, 200), (201, None)]


def head_share(counts: np.ndarray, head_fraction: float = 0.2) -> float:
    """Share of all interactions that go to the most popular ``head_fraction`` of items."""
    x = np.sort(np.asarray(counts))[::-1]
    k = max(1, round(head_fraction * x.size))
    return float(x[:k].sum() / x.sum())


def bucket_label(lo: int, hi: int | None) -> str:
    return f">{lo - 1}" if hi is None else f"{lo}-{hi}"


def activity_buckets(user_counts: pd.Series) -> dict[str, int]:
    out = {}
    for lo, hi in ACTIVITY_BUCKETS:
        mask = user_counts >= lo if hi is None else user_counts.between(lo, hi)
        out[bucket_label(lo, hi)] = int(mask.sum())
    return out


def _describe(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "min": int(s.min()),
        "p25": round(float(q[0.25]), 1),
        "median": round(float(q[0.5]), 1),
        "mean": round(float(s.mean()), 2),
        "p75": round(float(q[0.75]), 1),
        "p90": round(float(q[0.9]), 1),
        "p99": round(float(q[0.99]), 1),
        "max": int(s.max()),
    }


def summarize(ds: RecDataset, positive_threshold: float = 4.0) -> dict:
    inter = ds.interactions
    n_inter = len(inter)
    user_counts = inter.groupby("user_id").size()
    item_counts = inter.groupby("item_id").size()
    n_users, n_items = user_counts.size, item_counts.size
    density = n_inter / (n_users * n_items)

    positives = inter[inter["rating"] >= positive_threshold]
    pos_per_user = positives.groupby("user_id").size().reindex(user_counts.index, fill_value=0)

    # Chronology. Ties: several ratings by the same user with the identical second.
    dup_ts = inter.duplicated(["user_id", "timestamp"], keep=False)
    span_days = inter.groupby("user_id")["timestamp"].agg(lambda t: (t.max() - t.min()) / 86400)
    ts = pd.to_datetime(inter["timestamp"], unit="s")  # naive UTC
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    top3_month_share = float(monthly.sort_values(ascending=False).head(3).sum() / n_inter)

    return {
        "dataset": ds.name,
        "n_users": int(n_users),
        "n_items_rated": int(n_items),
        "n_items_catalog": len(ds.items),
        "n_interactions": int(n_inter),
        "density": round(density, 5),
        "sparsity_pct": round(100 * (1 - density), 2),
        "rating_mean": round(float(inter["rating"].mean()), 3),
        "rating_counts": {str(k): int(v) for k, v in inter["rating"].value_counts().sort_index().items()},
        "positive_threshold": positive_threshold,
        "n_positive": len(positives),
        "positive_share": round(len(positives) / n_inter, 4),
        "users_with_lt_10_positives": int((pos_per_user < 10).sum()),
        "users_with_lt_3_positives": int((pos_per_user < 3).sum()),
        "user_activity": _describe(user_counts),
        "user_activity_buckets": activity_buckets(user_counts),
        "item_popularity": _describe(item_counts),
        "items_with_lt_10_ratings": int((item_counts < 10).sum()),
        "top20pct_items_share_of_ratings": round(head_share(item_counts.to_numpy()), 4),
        "gini_item_popularity": round(gini(item_counts.to_numpy()), 4),
        "time_start": ts.min().date().isoformat(),
        "time_end": ts.max().date().isoformat(),
        "top3_months_share_of_ratings": round(top3_month_share, 4),
        "tied_timestamp_share": round(float(dup_ts.mean()), 4),
        "users_active_within_1_day_share": round(float((span_days <= 1).mean()), 4),
        "user_span_days_median": round(float(span_days.median()), 1),
    }


def genre_table(ds: RecDataset) -> pd.DataFrame:
    """Per genre: number of movies, number of ratings and mean rating."""
    exploded = ds.items[["item_id", "genres"]].explode("genres").rename(columns={"genres": "genre"})
    movies = exploded.groupby("genre").size().rename("movies")
    rated = ds.interactions.merge(exploded, on="item_id")
    agg = rated.groupby("genre")["rating"].agg(ratings="size", mean_rating="mean")
    return (
        pd.concat([movies, agg], axis=1)
        .fillna(0)
        .sort_values("ratings", ascending=False)
        .astype({"movies": int, "ratings": int})
    )
