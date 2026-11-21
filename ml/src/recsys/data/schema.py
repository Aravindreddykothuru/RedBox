"""The one data format every dataset is converted into.

Each loader (MovieLens, Amazon, ...) only has to produce a :class:`RecDataset`. Everything
downstream (splitting, models, evaluation) works on this schema and never on raw files,
so adding a dataset never touches model code.

interactions: user_id:int32, item_id:int32, rating:float32, timestamp:int64 (unix seconds)
items:        item_id:int32, title:str, year:Int16 (nullable), genres:list[str]
users:        user_id:int32 + dataset-specific columns (optional)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
INTERACTION_DTYPES = {"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"}


class SchemaError(ValueError):
    pass


@dataclass
class RecDataset:
    name: str
    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    def validate(self) -> RecDataset:
        """Fail fast on anything that would silently corrupt experiments."""
        inter = self.interactions
        missing = set(INTERACTION_COLUMNS) - set(inter.columns)
        if missing:
            raise SchemaError(f"interactions missing columns: {sorted(missing)}")
        if inter[INTERACTION_COLUMNS].isna().any().any():
            raise SchemaError("interactions contain nulls")
        dupes = inter.duplicated(["user_id", "item_id"]).sum()
        if dupes:
            raise SchemaError(f"{dupes} duplicate (user_id, item_id) pairs")
        unknown_items = ~inter["item_id"].isin(self.items["item_id"])
        if unknown_items.any():
            raise SchemaError(f"{unknown_items.sum()} interactions reference unknown item_ids")
        if self.items["item_id"].duplicated().any():
            raise SchemaError("items has duplicate item_ids")
        if self.users is not None:
            unknown_users = ~inter["user_id"].isin(self.users["user_id"])
            if unknown_users.any():
                raise SchemaError(f"{unknown_users.sum()} interactions reference unknown user_ids")
        if (inter["timestamp"] <= 0).any():
            raise SchemaError("non-positive timestamps")
        return self

    # ---------------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.interactions.to_parquet(directory / "interactions.parquet", index=False)
        self.items.to_parquet(directory / "items.parquet", index=False)
        if self.users is not None:
            self.users.to_parquet(directory / "users.parquet", index=False)
        meta = {"name": self.name, **self.meta}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> RecDataset:
        if not (directory / "interactions.parquet").exists():
            raise FileNotFoundError(
                f"No processed dataset at {directory}. Run `recsys prepare <dataset>` first."
            )
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        users_path = directory / "users.parquet"
        items = pd.read_parquet(directory / "items.parquet")
        if "genres" in items.columns:  # optional: Amazon rating files have no item metadata
            items["genres"] = items["genres"].map(list)  # parquet round-trips lists as arrays
        return cls(
            name=meta.pop("name"),
            interactions=pd.read_parquet(directory / "interactions.parquet"),
            items=items,
            users=pd.read_parquet(users_path) if users_path.exists() else None,
            meta=meta,
        )
