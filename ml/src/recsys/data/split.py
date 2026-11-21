"""Per-user chronological train / validation / test split.

Protocol (each choice is justified in notebooks/01_eda_movielens_1m.ipynb):

1. Keep "liked" interactions (rating >= ``positive_threshold``). ``None`` keeps every rating (ablation).
2. Order each user's interactions by time. Same-second ties (77% of ML-1M ratings, and they decide the
   split boundary for ~36% of users) are broken by a *seeded random key*: reproducible, but not tied to
   item ids or file order.
3. Per user: oldest interactions -> train, next ``val_ratio`` -> validation, newest ``test_ratio`` -> test,
   each part getting at least one interaction. Users with fewer than ``min_positives`` interactions are
   train-only: they still help learn item relationships but are not evaluated.
4. An item that never appears in train cannot be scored by any collaborative model, so its validation /
   test interactions are dropped (and counted in ``stats``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.schema import RecDataset

PARTS = ("train", "val", "test")


class SplitError(AssertionError):
    """The split violates a correctness guarantee (leakage, overlap, corrupted files)."""


@dataclass(frozen=True)
class SplitConfig:
    positive_threshold: float | None = 4.0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_positives: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        ratios_ok = 0 < self.val_ratio < 1 and 0 < self.test_ratio < 1
        if not ratios_ok or self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio and test_ratio must be in (0, 1) and sum to less than 1")
        if self.min_positives < 3:
            raise ValueError("min_positives must be >= 3 so train, validation and test each get an item")

    @property
    def name(self) -> str:
        thr = "all" if self.positive_threshold is None else f"pos{self.positive_threshold:g}"
        return f"{thr}_val{self.val_ratio:g}_test{self.test_ratio:g}_min{self.min_positives}_seed{self.seed}"


@dataclass
class TrainData:
    """Everything a model may use when fitting."""

    matrix: sp.csr_matrix  # (n_users, n_items), binary float32
    interactions: pd.DataFrame  # user_idx, item_idx, rating, timestamp; chronological within each user

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]


@dataclass
class Split:
    dataset: str
    config: SplitConfig
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user_ids: np.ndarray  # user_idx -> original user_id
    item_ids: np.ndarray  # item_idx -> original item_id
    stats: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    # ------------------------------------------------------------------ views
    def frame(self, *parts: str) -> pd.DataFrame:
        """Interactions of the given parts, chronological within each user."""
        unknown = set(parts) - set(PARTS)
        if unknown or not parts:
            raise ValueError(f"parts must be a non-empty subset of {PARTS}, got {parts}")
        df = pd.concat([getattr(self, p) for p in parts], ignore_index=True)
        # Each part is stored in time order and parts are concatenated oldest-first,
        # so a stable sort by user keeps every user's history chronological.
        return df.sort_values("user_idx", kind="stable").reset_index(drop=True)

    def matrix(self, *parts: str) -> sp.csr_matrix:
        df = self.frame(*parts)
        m = sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        m.sort_indices()
        return m

    def train_data(self, *parts: str) -> TrainData:
        parts = parts or ("train",)
        return TrainData(matrix=self.matrix(*parts), interactions=self.frame(*parts))

    def targets(self, part: str) -> sp.csr_matrix:
        return self.matrix(part)

    # ------------------------------------------------------------ integrity
    def fingerprint(self) -> str:
        """Short hash of the exact split; logged with every experiment for traceability."""
        h = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True).encode())
        for arr in (self.user_ids, self.item_ids):
            h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        for p in PARTS:
            df = getattr(self, p)
            for col in ("user_idx", "item_idx"):
                h.update(np.ascontiguousarray(df[col].to_numpy(), dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def check(self) -> None:
        """Raise SplitError if any leakage or consistency guarantee is broken."""
        keys = {}
        for p in PARTS:
            df = getattr(self, p)
            u, i = df["user_idx"].to_numpy(np.int64), df["item_idx"].to_numpy(np.int64)
            if len(df) and (u.min() < 0 or u.max() >= self.n_users or i.min() < 0 or i.max() >= self.n_items):
                raise SplitError(f"{p}: index out of range")
            keys[p] = u * self.n_items + i
            if np.unique(keys[p]).size != keys[p].size:
                raise SplitError(f"{p}: duplicate user-item pairs")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(keys[a], keys[b]).size:
                raise SplitError(f"{a} and {b} share user-item pairs (leakage)")

        spans = {}
        for p in PARTS:
            g = getattr(self, p).groupby("user_idx")["timestamp"]
            spans[p] = pd.DataFrame({f"{p}_min": g.min(), f"{p}_max": g.max()})
        for early, late in (("train", "val"), ("val", "test"), ("train", "test")):
            j = spans[early].join(spans[late], how="inner")
            if (j[f"{early}_max"] > j[f"{late}_min"]).any():
                raise SplitError(f"some {late} interaction is older than a {early} one (time leakage)")

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for p in PARTS:
            getattr(self, p).to_parquet(directory / f"{p}.parquet", index=False)
        pd.DataFrame({"user_id": self.user_ids}).to_parquet(directory / "users.parquet", index=False)
        pd.DataFrame({"item_id": self.item_ids}).to_parquet(directory / "items.parquet", index=False)
        meta = {"dataset": self.dataset, "config": asdict(self.config), "stats": self.stats}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Split:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        split = cls(
            dataset=meta["dataset"],
            config=SplitConfig(**meta["config"]),
            **{p: pd.read_parquet(directory / f"{p}.parquet") for p in PARTS},
            user_ids=pd.read_parquet(directory / "users.parquet")["user_id"].to_numpy(),
            item_ids=pd.read_parquet(directory / "items.parquet")["item_id"].to_numpy(),
            stats=meta["stats"],
        )
        if split.fingerprint() != meta["stats"]["fingerprint"]:
            raise SplitError(f"split files in {directory} do not match their recorded fingerprint")
        return split


def _round_half_up(x: np.ndarray) -> np.ndarray:
    # np.round uses banker's rounding (2.5 -> 2); split sizes should not depend on that quirk.
    return np.floor(x + 0.5).astype(np.int64)


def make_split(ds: RecDataset, config: SplitConfig | None = None) -> Split:
    config = config or SplitConfig()
    inter = ds.interactions[["user_id", "item_id", "rating", "timestamp"]]
    n_input = len(inter)
    if config.positive_threshold is not None:
        inter = inter[inter["rating"] >= config.positive_threshold]

    # Canonical row order first, so the random tie-break keys never depend on the input file's order.
    inter = inter.sort_values(["user_id", "item_id"], kind="stable").reset_index(drop=True)
    tie_key = np.random.default_rng(config.seed).random(len(inter))
    inter = (
        inter.assign(_tie=tie_key)
        .sort_values(["user_id", "timestamp", "_tie"], kind="stable")
        .reset_index(drop=True)
    )

    by_user = inter.groupby("user_id", sort=False)
    n = by_user["item_id"].transform("size").to_numpy()
    pos = by_user.cumcount().to_numpy()
    n_test = np.maximum(1, _round_half_up(n * config.test_ratio))
    n_val = np.maximum(1, _round_half_up(n * config.val_ratio))
    evaluable = (n >= config.min_positives) & (n - n_test - n_val >= 1)

    part = np.zeros(len(inter), dtype=np.int8)  # 0 train, 1 val, 2 test
    part[evaluable & (pos >= n - n_test - n_val)] = 1
    part[evaluable & (pos >= n - n_test)] = 2
    inter["_part"] = part

    train_raw = inter[part == 0]
    user_ids = np.sort(train_raw["user_id"].unique())  # every kept user has >= 1 train interaction
    item_ids = np.sort(train_raw["item_id"].unique())
    user_index, item_index = pd.Index(user_ids), pd.Index(item_ids)

    frames, cold = {}, {}
    for code, name in enumerate(PARTS):
        raw = inter[part == code]
        enc = pd.DataFrame(
            {
                "user_idx": user_index.get_indexer(raw["user_id"]).astype(np.int32),
                "item_idx": item_index.get_indexer(raw["item_id"]).astype(np.int32),
                "rating": raw["rating"].to_numpy(np.float32),
                "timestamp": raw["timestamp"].to_numpy(np.int64),
            }
        )
        known = enc["item_idx"].to_numpy() >= 0
        cold[name] = int((~known).sum())
        frames[name] = enc[known].reset_index(drop=True)

    # How often did the random tie-break (not time) decide the boundary between parts?
    spans = inter[evaluable].pivot_table(
        index="user_id", columns="_part", values="timestamp", aggfunc=["min", "max"]
    )
    boundary_tie = (spans[("max", 0)] == spans[("min", 1)]) | (spans[("max", 1)] == spans[("min", 2)])

    stats = {
        "n_input_interactions": n_input,
        "n_kept_interactions": len(inter),
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_users_dropped_no_positives": int(ds.interactions["user_id"].nunique() - len(user_ids)),
        "n_train_only_users": int(len(user_ids) - inter.loc[evaluable, "user_id"].nunique()),
        "interactions": {p: len(frames[p]) for p in PARTS},
        "users_with_targets": {p: int(frames[p]["user_idx"].nunique()) for p in ("val", "test")},
        "cold_item_interactions_dropped": {p: cold[p] for p in ("val", "test")},
        "boundary_decided_by_tie_break_share": round(float(boundary_tie.mean()), 4),
    }
    split = Split(dataset=ds.name, config=config, user_ids=user_ids, item_ids=item_ids, stats=stats, **frames)
    split.stats["fingerprint"] = split.fingerprint()
    return split
