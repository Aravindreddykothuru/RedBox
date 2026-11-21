"""Optuna tuning on the validation set, with the same time budget for every model.

Fairness rules (they answer RQ1 honestly: classic models must be tuned as carefully as deep ones):
  * every model gets the same wall-clock budget (and the same trial cap); fast models reach the cap,
    slow models get fewer trials, which is itself a real cost of those models;
  * trial 0 always uses the paper defaults, so tuning can never make a model look worse on validation;
  * only validation data is used; the test set is touched once, later, by the normal benchmark protocol;
  * iterative models report every validation score, so clearly bad trials are stopped early (median
    pruning), and a trial that runs past the budget is stopped at its next validation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import optuna

from recsys.data.split import Split
from recsys.eval.evaluator import evaluate
from recsys.experiments.config import ExperimentConfig
from recsys.models import build, get_model_class
from recsys.paths import mlruns_dir
from recsys.tuning.spaces import SPACES

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class _Recorder:
    """A stand-in trial that records which parameters a search space asks for, and their ranges."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.distributions: dict[str, optuna.distributions.BaseDistribution] = {}

    def _rec(self, name: str, value, dist):
        self.names.append(name)
        self.distributions[name] = dist
        return value

    def suggest_int(self, name, low, high, log=False, step=1):
        return self._rec(name, low, optuna.distributions.IntDistribution(low, high, log=log, step=step))

    def suggest_float(self, name, low, high, log=False, step=None):
        return self._rec(name, low, optuna.distributions.FloatDistribution(low, high, log=log, step=step))

    def suggest_categorical(self, name, choices):
        return self._rec(name, choices[0], optuna.distributions.CategoricalDistribution(choices))


def seed_default_from_mlflow(
    study: optuna.Study, cfg, split: Split, entry: str, params: dict, class_name: str
) -> bool:
    """If `recsys benchmark` already ran these exact default settings, record its validation score as
    trial 0 instead of training the same model again (saves about an hour for SASRec or LightGCN)."""
    from recsys.experiments.runner import find_run, params_hash
    from recsys.experiments.tracking import metric_key

    run = find_run(cfg.dataset, split.stats["fingerprint"], entry, params_hash(params))
    value = run.data.metrics.get(f"val_{metric_key(cfg.primary_metric)}") if run else None
    if value is None:
        return False
    base = {k: v for k, v in params.items() if k != "model"}
    rec = _Recorder()
    SPACES[class_name](rec)  # type: ignore[arg-type]
    attrs = {"seeded_from_mlflow_run": run.info.run_id}
    if "model.epochs" in run.data.params:
        attrs["best_epoch"] = int(run.data.params["model.epochs"])
    study.add_trial(
        optuna.trial.create_trial(
            params=default_trial_params(class_name, base),
            distributions=rec.distributions,
            value=value,
            user_attrs=attrs,
        )
    )
    log.info("%s: trial 0 (defaults) taken from MLflow run %s: val %.4f", entry, run.info.run_id[:8], value)
    return True


def default_trial_params(model_class: str, base: dict[str, Any]) -> dict[str, Any]:
    """The search-space values that reproduce the model's default configuration (enqueued as trial 0)."""
    rec = _Recorder()
    SPACES[model_class](rec)  # type: ignore[arg-type]
    defaults = {**get_model_class(model_class).defaults, **base}
    out = {}
    for name in rec.names:
        value = defaults[name]
        out[name] = json.dumps(value) if isinstance(value, list) else value
    return out


def storage_url() -> str:
    mlruns_dir().mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + (mlruns_dir() / "optuna.db").as_posix()


def study_name(dataset: str, fingerprint: str, entry: str) -> str:
    return f"{dataset}:{fingerprint}:{entry}"


@dataclass
class TuningResult:
    entry: str
    model_class: str
    n_complete: int
    n_pruned: int
    default_value: float | None
    best_value: float | None
    best_params: dict[str, Any]  # full parameter set for the config file
    seconds: float


def tune_entry(
    entry: str,
    base_params: dict[str, Any],
    split: Split,
    cfg: ExperimentConfig,
    *,
    minutes: float,
    max_trials: int,
    seed: int = 0,
) -> TuningResult:
    """Tune one config entry (e.g. "sasrec_ce") on the validation set."""
    base = dict(base_params)
    class_name = base.pop("model", entry)
    model_cls = get_model_class(class_name)
    space = SPACES[class_name]
    train = split.train_data("train")
    val_targets = split.targets("val")
    k = cfg.primary_k
    deadline = time.monotonic() + minutes * 60

    def score(model) -> float:
        return evaluate(model, train.matrix, val_targets, ks=(k,)).metrics[cfg.primary_metric]

    def objective(trial: optuna.Trial) -> float:
        params = {**base, **space(trial)}
        model = build(class_name, **params)
        if not model_cls.iterative:
            model.fit(train)
            return score(model)

        first_trial = trial.number == 0

        def validator(m) -> float:
            value = score(m)
            epoch = len(m.history_) + 1  # history is appended after validation
            trial.report(value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
            if not first_trial and time.monotonic() > deadline:
                raise optuna.TrialPruned("time budget exhausted")
            return value

        model.fit(train, validator=validator)
        trial.set_user_attr("best_epoch", int(model.best_epoch_))
        return max(r["val"] for r in model.history_ if "val" in r)

    eval_every = int({**model_cls.defaults, **base}.get("eval_every", 1))
    study = optuna.create_study(
        study_name=study_name(cfg.dataset, split.stats["fingerprint"], entry),
        storage=storage_url(),
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3 * eval_every),
    )
    already = len(study.trials)
    if already == 0:
        if seed_default_from_mlflow(study, cfg, split, entry, base_params, class_name):
            already = 1
        else:
            study.enqueue_trial(default_trial_params(class_name, base))
    remaining = max(max_trials - already, 0)
    t0 = time.perf_counter()
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=minutes * 60,
            catch=(MemoryError,),
            callbacks=[_log_trial(entry)],
        )
    seconds = time.perf_counter() - t0
    return summarize(study, entry, class_name, base, seconds)


def _log_trial(entry: str):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"{trial.value:.4f}"
        best = (
            study.best_value
            if any(t.state.is_finished() and t.value is not None for t in study.trials)
            else None
        )
        log.info(
            "%s trial %3d  %-8s best %s  %s",
            entry,
            trial.number,
            value,
            f"{best:.4f}" if best else "-",
            trial.params,
        )

    return callback


def summarize(study: optuna.Study, entry: str, class_name: str, base: dict, seconds: float) -> TuningResult:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    default = next((t for t in complete if t.number == 0), None)
    best = max(complete, key=lambda t: t.value) if complete else None
    params = dict(base)
    if best is not None:
        for name, value in best.params.items():
            params[name] = json.loads(value) if name == "mlp_dims" else value
    if class_name != entry:  # e.g. sasrec_ce -> model: sasrec
        params = {"model": class_name, **params}
    return TuningResult(
        entry=entry,
        model_class=class_name,
        n_complete=len(complete),
        n_pruned=len(pruned),
        default_value=default.value if default else None,
        best_value=best.value if best else None,
        best_params=params,
        seconds=seconds,
    )


def param_importance(study: optuna.Study) -> dict[str, float]:
    """Which hyperparameters mattered most (PED-ANOVA; needs a handful of finished trials)."""
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 8 or len(complete[0].params) < 2:
        return {}
    try:
        evaluator = optuna.importance.PedAnovaImportanceEvaluator()
        return {
            k: float(v)
            for k, v in optuna.importance.get_param_importances(study, evaluator=evaluator).items()
        }
    except (RuntimeError, ValueError):  # too few distinct values for the quantile split
        return {}


def load_study(dataset: str, fingerprint: str, entry: str) -> optuna.Study | None:
    try:
        return optuna.load_study(study_name=study_name(dataset, fingerprint, entry), storage=storage_url())
    except KeyError:
        return None
