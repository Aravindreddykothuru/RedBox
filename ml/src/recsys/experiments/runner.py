"""The single experiment protocol shared by every model.

    1. fit on TRAIN (iterative models early-stop on VALIDATION NDCG)  -> validation metrics
    2. refit on TRAIN + VALIDATION with the chosen settings              -> test metrics

The test set is touched exactly once per model, after all choices are fixed, so test numbers are
an honest estimate of performance on unseen future interactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import scipy.sparse as sp

from recsys.data.features import genre_matrix
from recsys.data.schema import RecDataset
from recsys.data.split import Split, SplitConfig, make_split
from recsys.eval.evaluator import EvalResult, evaluate
from recsys.experiments.config import ExperimentConfig
from recsys.models import build, get_model_class
from recsys.paths import models_dir, processed_dir, run_outputs_dir, splits_dir

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    model: str  # run name = config entry, e.g. "sasrec_ce"
    params: dict[str, Any]  # parameters of the final (refit) model
    val: EvalResult
    test: EvalResult
    fit_seconds: float
    refit_seconds: float
    curve: list[dict] = field(default_factory=list)  # training curve of iterative models
    model_class: str = ""  # registry name, e.g. "sasrec"
    params_hash: str = ""  # identifies the exact config settings (see params_hash())


def params_hash(params: dict[str, Any]) -> str:
    """Short hash of a config entry's settings. Runs and saved models are keyed by it, so a run with
    identical settings (e.g. a model whose tuned parameters equal its defaults) is never repeated."""
    text = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_or_create_split(dataset: str, split_config: SplitConfig) -> Split:
    """Load a saved split, or create, leakage-check and save it on first use. A split older than the
    processed data (e.g. after app ratings were imported) is rebuilt, so it never describes stale data;
    its new fingerprint then makes every model retrain."""
    directory = splits_dir(dataset, split_config.name)
    data = processed_dir(dataset) / "interactions.parquet"
    if (directory / "meta.json").exists():
        if not data.exists() or (directory / "meta.json").stat().st_mtime >= data.stat().st_mtime:
            return Split.load(directory)
        log.warning("%s: the data changed after the split was made; splitting again", dataset)
    split = make_split(RecDataset.load(processed_dir(dataset)), split_config)
    split.check()
    split.save(directory)
    return split


def load_split(cfg: ExperimentConfig) -> Split:
    """The split configured for this experiment."""
    return load_or_create_split(cfg.dataset, cfg.split)


def load_item_features(split: Split) -> sp.csr_matrix | None:
    ds = RecDataset.load(processed_dir(split.dataset))
    result = genre_matrix(ds.items, split.item_ids)
    return None if result is None else result[0]


def model_path(split: Split, entry: str, settings_hash: str, phase: str) -> Path:
    """Where the fitted model of one config entry is saved: phase "train" (for building the hybrid's
    ranker data) or "trainval" (the final model evaluated on test, and reused by the hybrid)."""
    return models_dir(split.dataset, split.config.name, f"{entry}-{settings_hash}_{phase}.pkl")


def models_saved(split: Split, entry: str, settings_hash: str) -> bool:
    return all(model_path(split, entry, settings_hash, p).exists() for p in ("train", "trainval"))


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_model(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def run_model(
    name: str,
    params: dict[str, Any],
    split: Split,
    cfg: ExperimentConfig,
    item_features: sp.csr_matrix | None = None,
    save_models: bool = False,
) -> RunResult:
    """Run config entry ``name``. Its params may contain ``model: <registry name>`` so one model class
    can appear several times with different settings (e.g. sasrec and sasrec_ce)."""
    settings_hash = params_hash(params)
    params = dict(params)
    class_name = params.pop("model", name)
    model_cls = get_model_class(class_name)
    train = split.train_data("train")
    val_targets = split.targets("val")

    def validator(model) -> float:
        res = evaluate(model, train.matrix, val_targets, ks=(cfg.primary_k,))
        return res.metrics[cfg.primary_metric]

    # Phase 1: train -> validation
    model = build(class_name, **params)
    t0 = time.perf_counter()
    model.fit(train, validator=validator if model_cls.iterative else None)
    fit_seconds = time.perf_counter() - t0
    val = evaluate(model, train.matrix, val_targets, ks=cfg.ks, item_features=item_features)
    if save_models:
        save_model(model, model_path(split, name, settings_hash, "train"))
    log.info(
        "%-8s val  %s=%.4f  (fit %.1fs)",
        name,
        cfg.primary_metric,
        val.metrics[cfg.primary_metric],
        fit_seconds,
    )

    # Phase 2: refit on train+validation with frozen settings -> test (touched once)
    final_params = {**model.params, **model.refit_params()}
    trainval = split.train_data("train", "val")
    final = build(class_name, **final_params)
    t0 = time.perf_counter()
    final.fit(trainval)
    refit_seconds = time.perf_counter() - t0
    test = evaluate(final, trainval.matrix, split.targets("test"), ks=cfg.ks, item_features=item_features)
    if save_models:
        save_model(final, model_path(split, name, settings_hash, "trainval"))
    log.info(
        "%-8s test %s=%.4f  (refit %.1fs)",
        name,
        cfg.primary_metric,
        test.metrics[cfg.primary_metric],
        refit_seconds,
    )

    return RunResult(
        model=name,
        params=final_params,
        val=val,
        test=test,
        fit_seconds=fit_seconds,
        refit_seconds=refit_seconds,
        curve=list(getattr(model, "history_", [])),
        model_class=class_name,
        params_hash=settings_hash,
    )


def find_run(dataset: str, fingerprint: str, entry: str, settings_hash: str):
    """Latest MLflow run of ``entry`` with exactly these settings on this split, or None."""
    from recsys.experiments import tracking

    mlflow = tracking.start(dataset)
    exp = mlflow.get_experiment_by_name(dataset)
    runs = mlflow.tracking.MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.model = '{entry}' and tags.split_fingerprint = '{fingerprint}' "
            f"and tags.params_hash = '{settings_hash}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def save_per_user(result: RunResult, split: Split, untracked: bool = False) -> dict[str, str]:
    """Per-user results for significance tests. Runs not logged to MLflow (quick experiments) go to an
    ``untracked/`` subfolder so they can never be mistaken for the results behind the report tables."""
    out = run_outputs_dir(split.dataset, split.config.name, *(["untracked"] if untracked else []))
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in ("val", "test"):
        path = out / f"{result.model}_{phase}.parquet"
        getattr(result, phase).per_user.to_parquet(path, index=False)
        paths[phase] = str(path)
    return paths


def log_to_mlflow(
    result: RunResult, split: Split, cfg: ExperimentConfig, per_user_paths: dict[str, str]
) -> str:
    from recsys.experiments import tracking

    mlflow = tracking.start(f"{cfg.dataset}")
    with mlflow.start_run(run_name=result.model) as run:
        mlflow.set_tags(
            {
                "model": result.model,
                "model_class": result.model_class or result.model,
                "dataset": cfg.dataset,
                "split": split.config.name,
                "split_fingerprint": split.stats["fingerprint"],
                "config": str(cfg.source),
                "params_hash": result.params_hash,
            }
        )
        mlflow.log_params({f"model.{k}": v for k, v in result.params.items()})
        mlflow.log_params({f"split.{k}": v for k, v in vars(split.config).items()})
        metrics = {f"val_{tracking.metric_key(k)}": v for k, v in result.val.metrics.items()}
        metrics |= {f"test_{tracking.metric_key(k)}": v for k, v in result.test.metrics.items()}
        metrics |= {
            "fit_seconds": result.fit_seconds,
            "refit_seconds": result.refit_seconds,
            # lazy models (UserKNN) do their work at scoring time, so evaluation time matters too
            "val_eval_seconds": result.val.seconds,
            "test_eval_seconds": result.test.seconds,
        }
        mlflow.log_metrics(metrics)
        for point in result.curve:
            mlflow.log_metric("train_loss", point["loss"], step=point["epoch"])
            if "val" in point:
                mlflow.log_metric(
                    f"val_curve_{tracking.metric_key(cfg.primary_metric)}", point["val"], step=point["epoch"]
                )
        for path in per_user_paths.values():
            mlflow.log_artifact(path, artifact_path="per_user")
        return run.info.run_id
