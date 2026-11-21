"""The hybrid's experiment protocol (no leakage, test touched once).

    generators fitted on TRAIN  -> candidates + features for each user -> labels = VALIDATION items
                                -> LightGBM LambdaRank ranker (85% of users; 15% held out for early stopping
                                   and for the hybrid's validation score)
    generators fitted on TRAIN+VALIDATION -> candidates + features -> ranker -> evaluated once on TEST

The ranker learns which generator to trust in which situation (e.g. sequence models right after a
burst of ratings, taste models for users with long histories) from data the test period never saw.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.schema import RecDataset
from recsys.data.split import Split
from recsys.eval.evaluator import EvalResult, evaluate
from recsys.experiments.config import ExperimentConfig
from recsys.experiments.runner import RunResult, load_model, model_path
from recsys.hybrid.features import Context, build_features, last_items
from recsys.hybrid.ranker import HybridScores, score_matrix, train_ranker
from recsys.models.base import Recommender
from recsys.paths import processed_dir

log = logging.getLogger(__name__)
ModelLoader = Callable[[str], dict[str, Recommender]]  # phase ("train" | "trainval") -> fitted generators


def disk_loader(split: Split, generators: dict[str, tuple[str, str]]) -> ModelLoader:
    """Generators saved by `recsys benchmark` (name -> (config entry, settings hash)), so slow models
    such as SASRec are never retrained just for the hybrid."""

    def load(phase: str) -> dict[str, Recommender]:
        out = {}
        for name, (entry, settings_hash) in generators.items():
            path = model_path(split, entry, settings_hash, phase)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run `recsys benchmark --models {entry}` first")
            out[name] = load_model(path)
        return out

    return load


def side_information(split: Split) -> tuple[sp.csr_matrix | None, np.ndarray | None, pd.DataFrame | None]:
    """Item genres, item release years and user demographics aligned to the split's indices."""
    from recsys.data.features import genre_matrix

    ds = RecDataset.load(processed_dir(split.dataset))
    gm = genre_matrix(ds.items, split.item_ids)
    years = None
    if "year" in ds.items.columns:
        years = ds.items.set_index("item_id")["year"].reindex(split.item_ids).astype("float64").to_numpy()
    side = None
    if ds.users is not None:
        u = ds.users.set_index("user_id").reindex(split.user_ids)
        cols = {}
        if "gender" in u:
            cols["is_female"] = (u["gender"] == "F").astype(float)
        for c in ("age_code", "occupation_code"):
            if c in u:
                cols[c] = u[c].astype(float)
        side = pd.DataFrame(cols).set_axis(np.arange(len(split.user_ids)), axis=0) if cols else None
    return (gm[0] if gm else None), years, side


def _users_with(targets: sp.csr_matrix) -> np.ndarray:
    return np.flatnonzero(np.diff(targets.tocsr().indptr) > 0)


def _restrict_rows(targets: sp.csr_matrix, users: np.ndarray) -> sp.csr_matrix:
    keep = np.zeros(targets.shape[0])
    keep[users] = 1.0
    return (sp.diags(keep) @ targets).tocsr()


def run_hybrid(
    cfg: ExperimentConfig, split: Split, load_models: ModelLoader, item_features: sp.csr_matrix | None
) -> tuple[list[RunResult], dict[str, Any]]:
    h = cfg.hybrid
    n_cand = int(h.get("candidates_per_generator", 40))
    lambdas = [float(x) for x in h.get("mmr_lambdas", [1.0])]
    seed = int(h.get("seed", 42))
    genres, years, side = side_information(split)
    genres_dense = genres.toarray() if genres is not None else None
    shape = (split.n_users, split.n_items)

    # ---- learn the ranker: generators fitted on train, labels from validation
    t0 = time.perf_counter()
    train = split.train_data("train")
    val_targets = split.targets("val")
    ctx = Context(train.matrix, last_items(train.interactions, split.n_users), genres, years, side)
    val_users = _users_with(val_targets)
    val_table = build_features(load_models("train"), val_users, ctx, n_candidates=n_cand, targets=val_targets)
    log.info(
        "hybrid: %d candidate pairs, candidate recall on validation %.3f",
        len(val_table.users),
        val_table.candidate_recall,
    )
    booster, info = train_ranker(val_table, params=h.get("ranker"), seed=seed)
    fit_seconds = time.perf_counter() - t0
    held = info.pop("held_users")
    log.info("hybrid ranker: %s", info)

    # ---- apply once to the test period: generators refitted on train+validation
    t0 = time.perf_counter()
    trainval = split.train_data("train", "val")
    test_targets = split.targets("test")
    ctx_tv = Context(trainval.matrix, last_items(trainval.interactions, split.n_users), genres, years, side)
    test_table = build_features(
        load_models("trainval"), _users_with(test_targets), ctx_tv, n_candidates=n_cand, targets=test_targets
    )
    apply_seconds = time.perf_counter() - t0

    held_val = _restrict_rows(val_targets, held)
    results = []
    for lam in lambdas:
        val_scores = score_matrix(val_table, booster, shape, lam=lam, genres=genres_dense, k=max(cfg.ks))
        test_scores = score_matrix(test_table, booster, shape, lam=lam, genres=genres_dense, k=max(cfg.ks))
        val = evaluate(
            HybridScores(val_scores), train.matrix, held_val, ks=cfg.ks, item_features=item_features
        )
        test = evaluate(
            HybridScores(test_scores), trainval.matrix, test_targets, ks=cfg.ks, item_features=item_features
        )
        name = "hybrid" if lam == 1.0 else f"hybrid_mmr{lam:g}"
        m = test.metrics
        log.info("%-14s test %s=%.4f  coverage@10=%.3f  ild@10=%.3f", name, cfg.primary_metric,
                 m[cfg.primary_metric], m["coverage@10"], m.get("ild@10", np.nan))  # fmt: skip
        params = {
            "generators": ",".join(h["generators"]),
            "candidates_per_generator": n_cand,
            "mmr_lambda": lam,
            "ranker_best_iteration": info["best_iteration"],
            **(h.get("ranker") or {}),
        }
        results.append(
            RunResult(name, params, val, test, fit_seconds, apply_seconds, curve=[], model_class="hybrid")
        )

    importance = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    ).sort_values(ascending=False)
    details = {
        **info,
        "candidate_recall_val": val_table.candidate_recall,
        "candidate_recall_test": test_table.candidate_recall,
        "importance": importance / importance.sum(),
        "booster": booster,
    }
    return results, details


__all__ = ["EvalResult", "disk_loader", "run_hybrid"]
