"""Filesystem locations used across the project.

Everything heavy (raw downloads, processed parquet, model artifacts, MLflow runs) lives
outside git. The data root defaults to ``<repo>/data`` and can be moved with the
``RECSYS_DATA_DIR`` environment variable (e.g. to a bigger disk or a Kaggle volume).
"""

from __future__ import annotations

import os
from pathlib import Path

# ml/src/recsys/paths.py -> parents[3] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return Path(os.environ.get("RECSYS_DATA_DIR", REPO_ROOT / "data"))


def raw_dir(dataset: str | None = None) -> Path:
    base = data_dir() / "raw"
    return base / dataset if dataset else base


def processed_dir(dataset: str | None = None) -> Path:
    base = data_dir() / "processed"
    return base / dataset if dataset else base


def splits_dir(*parts: str) -> Path:
    return data_dir().joinpath("splits", *parts)


def models_dir(*parts: str) -> Path:
    """Fitted models (pickles), reused by the hybrid and exported for serving; git-ignored."""
    return data_dir().joinpath("models", *parts)


def run_outputs_dir(*parts: str) -> Path:
    """Per-user results of every run (inputs to significance tests); git-ignored."""
    return data_dir().joinpath("results", *parts)


def mlruns_dir() -> Path:
    return REPO_ROOT / "mlruns"


def configs_dir() -> Path:
    return REPO_ROOT / "ml" / "configs"


def figures_dir(*parts: str) -> Path:
    """Report-ready figures are versioned in git under docs/figures."""
    return REPO_ROOT.joinpath("docs", "figures", *parts)


def results_docs_dir(*parts: str) -> Path:
    """Report-ready result tables, versioned in git under docs/results."""
    return REPO_ROOT.joinpath("docs", "results", *parts)
