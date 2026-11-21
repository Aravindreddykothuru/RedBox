"""MLflow tracking: every run's parameters, metrics and per-user results are recorded.

Runs are stored in <repo>/mlruns (git-ignored). Browse them with:
    uv run mlflow ui --backend-store-uri "sqlite:///../mlruns/mlflow.db"
"""

from __future__ import annotations

import logging
import os
import re

from recsys.paths import mlruns_dir

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
for noisy in ("alembic", "mlflow"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def tracking_uri() -> str:
    return "sqlite:///" + (mlruns_dir() / "mlflow.db").as_posix()


def start(experiment: str):
    """Point MLflow at the project store and select ``experiment``. Returns the mlflow module."""
    import mlflow

    root = mlruns_dir()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        exp_id = mlflow.create_experiment(
            experiment, artifact_location=(root / "artifacts" / experiment).as_uri()
        )
    else:
        exp_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return mlflow


def metric_key(name: str) -> str:
    """MLflow metric names cannot contain '@': ndcg@10 -> ndcg_at_10."""
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", name.replace("@", "_at_"))
