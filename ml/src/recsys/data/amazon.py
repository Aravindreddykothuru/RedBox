"""Parser for Amazon Reviews 2023 "5-core rating_only" files (Hou et al., 2024).

The official 5-core keeps users and items with at least 5 reviews of any rating. We then apply an
iterative k-core on *liked* reviews (rating >= 4): items liked by at least ``min_item`` users and users
with at least ``min_user`` liked items, repeated until stable. This keeps every user splittable into
train/validation/test and keeps the item count small enough for EASE's item x item matrix on a laptop
(5,771 items instead of 25,612). All ratings of the remaining users and items are kept; the split later
applies the same "liked" threshold as for MovieLens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.data.schema import INTERACTION_DTYPES, RecDataset


def iterative_kcore(pairs: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Repeatedly drop items with < min_item rows and users with < min_user rows until nothing changes."""
    while True:
        item_counts = pairs.groupby("item")["user"].transform("size")
        kept = pairs[item_counts >= min_item]
        user_counts = kept.groupby("user")["item"].transform("size")
        kept = kept[user_counts >= min_user]
        if len(kept) == len(pairs):
            return kept
        pairs = kept


def load_amazon(
    path: Path, name: str, *, min_user: int = 5, min_item: int = 15, positive_threshold: float = 4.0
) -> RecDataset:
    raw = pd.read_csv(path)  # user_id, parent_asin, rating, timestamp (milliseconds)
    raw = raw.rename(columns={"user_id": "user", "parent_asin": "item"})
    liked = raw.loc[raw["rating"] >= positive_threshold, ["user", "item"]].drop_duplicates()
    core = iterative_kcore(liked, min_user, min_item)
    users = pd.Index(sorted(core["user"].unique()))
    items = pd.Index(sorted(core["item"].unique()))
    keep = raw[raw["user"].isin(users) & raw["item"].isin(items)]
    # a user may review the same product twice: keep the latest review
    keep = keep.sort_values("timestamp").drop_duplicates(["user", "item"], keep="last")

    interactions = pd.DataFrame(
        {
            "user_id": users.get_indexer(keep["user"]) + 1,
            "item_id": items.get_indexer(keep["item"]) + 1,
            "rating": keep["rating"].to_numpy(),
            "timestamp": keep["timestamp"].to_numpy() // 1000,  # ms -> s (schema)
        }
    ).astype(INTERACTION_DTYPES)
    item_table = pd.DataFrame({"item_id": (pd.RangeIndex(len(items)) + 1).astype("int32"), "title": items})
    return RecDataset(
        name=name,
        interactions=interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True),
        items=item_table,
        users=None,
        meta={
            "rating_scale": [1, 5],
            "source": "https://amazon-reviews-2023.github.io/ (5-core, rating_only)",
            "kcore": {"min_user": min_user, "min_item": min_item, "on": f"rating >= {positive_threshold}"},
        },
    ).validate()


def load_amazon_videogames(directory: Path) -> RecDataset:
    return load_amazon(directory / "Video_Games.csv.gz", "amazon-videogames")
