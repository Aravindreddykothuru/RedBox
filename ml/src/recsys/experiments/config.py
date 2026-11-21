"""Experiment configuration loaded from ml/configs/<name>.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from recsys.data.split import SplitConfig
from recsys.paths import configs_dir


@dataclass
class ExperimentConfig:
    dataset: str
    split: SplitConfig
    ks: tuple[int, ...] = (5, 10, 20)
    primary_metric: str = "ndcg@10"
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: Path | None = None
    description: str = ""  # shown in the report tables (e.g. "literature defaults, not tuned")
    hybrid: dict[str, Any] = field(default_factory=dict)  # Red Box settings (generators, ranker, MMR)
    serving: dict[str, Any] = field(default_factory=dict)  # the app's hybrid: only history-based generators

    @property
    def primary_k(self) -> int:
        return int(self.primary_metric.split("@")[1])

    @classmethod
    def load(cls, name_or_path: str | Path) -> ExperimentConfig:
        path = Path(name_or_path)
        if not path.suffix:  # a bare name like "ml-1m"
            path = configs_dir() / f"{name_or_path}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        evaluation = raw.get("evaluation", {})
        cfg = cls(
            dataset=raw["dataset"],
            split=SplitConfig(**raw.get("split", {})),
            ks=tuple(evaluation.get("ks", (5, 10, 20))),
            primary_metric=evaluation.get("primary_metric", "ndcg@10"),
            models={k: dict(v or {}) for k, v in raw.get("models", {}).items()},
            source=path,
            description=raw.get("description", ""),
            hybrid=dict(raw.get("hybrid") or {}),
            serving=dict(raw.get("serving") or {}),
        )
        if cfg.primary_k not in cfg.ks:
            raise ValueError(f"primary metric {cfg.primary_metric} must use one of ks={cfg.ks}")
        return cfg
