"""Export everything the API needs into one folder (the "bundle"): NumPy arrays, the LightGBM ranker
and item metadata. No PyTorch and no pickles are needed to load it.

The serving ranker is trained exactly like the offline hybrid (generators fitted on train, labels from
validation), but its features come from the serving scorers themselves, so training-time and
request-time features are computed by identical code. It is then evaluated once on the test set, and
that score is recorded in the manifest as the quality of the deployed model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.schema import RecDataset
from recsys.data.split import Split
from recsys.eval.evaluator import evaluate
from recsys.experiments.config import ExperimentConfig
from recsys.experiments.runner import load_model, model_path, params_hash
from recsys.hybrid.features import Context, build_features, item_cosine
from recsys.hybrid.ranker import HybridScores, score_matrix, train_ranker
from recsys.paths import processed_dir, results_docs_dir
from recsys.serving.scorers import (
    EASEScorer,
    HistoryScorer,
    IALSScorer,
    MarkovScorer,
    PopularityScorer,
    SASRecScorer,
)

log = logging.getLogger(__name__)
BUNDLE_VERSION = 1
NEIGHBOURS = 50  # stored similar items per item (explanations, "similar movies")


def sequences_of(log_df: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """Chronological item sequence of every user (empty array for users without history)."""
    groups = log_df.groupby("user_idx", sort=True)["item_idx"].apply(lambda s: s.to_numpy(np.int64))
    return [groups.get(u, np.empty(0, dtype=np.int64)) for u in range(n_users)]


def scorers_from_models(models: dict[str, object], classes: dict[str, str]) -> dict[str, HistoryScorer]:
    """Turn fitted training models into torch-free history scorers (keyed by generator name)."""
    out: dict[str, HistoryScorer] = {}
    for name, m in models.items():
        cls = classes[name]
        if cls == "ease":
            out[name] = EASEScorer(m.B_)
        elif cls == "ials":
            out[name] = IALSScorer(m.item_factors_, m.params["alpha"], m.params["reg"])
        elif cls == "markov":
            out[name] = MarkovScorer(m.transitions_, m.popularity_, m.params["pop_weight"])
        elif cls == "mostpop":
            out[name] = PopularityScorer(m.popularity_)
        elif cls == "sasrec":
            cfg = {k: m.params[k] for k in ("dim", "heads", "blocks", "max_len")}
            out[name] = SASRecScorer.from_torch(m.net_, cfg)
        else:
            raise ValueError(f"{name} ({cls}) cannot be served from a user's history")
    return out


def _attach(scorers: dict[str, HistoryScorer], history: sp.csr_matrix, seqs: list[np.ndarray]) -> None:
    for s in scorers.values():
        s.set_history(history, seqs)


def item_neighbours(history: sp.csr_matrix, k: int = NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours of every item (indices, similarities), self excluded."""
    sim = item_cosine(history)
    np.fill_diagonal(sim, -1.0)
    k = min(k, sim.shape[0] - 1)
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int32), np.take_along_axis(vals, order, axis=1)


def onboarding_seeds(ds: RecDataset, item_ids: np.ndarray, n: int = 60) -> np.ndarray:
    """Movies worth asking a new user about: well known *and* divisive (high rating spread), because
    those ratings tell the most about a person's taste. Returned as item indices of the split."""
    stats = ds.interactions.groupby("item_id")["rating"].agg(["size", "std"]).reindex(item_ids)
    informativeness = np.log1p(stats["size"].fillna(0)) * stats["std"].fillna(0)
    return np.argsort(-informativeness.to_numpy())[:n].astype(np.int32)


def _sample_users(users: np.ndarray, limit: int | None, rng: np.random.Generator) -> np.ndarray:
    """At most ``limit`` users, drawn without replacement and kept in index order (all if no limit)."""
    if not limit or users.size <= int(limit):
        return users
    return np.sort(rng.choice(users, size=int(limit), replace=False))


@dataclass
class ExportReport:
    path: Path
    test_metrics: dict[str, float]
    ranker_info: dict
    seconds: float


def export_bundle(cfg: ExperimentConfig, split: Split, out_dir: Path) -> ExportReport:
    t0 = time.perf_counter()
    serving = cfg.serving
    generators: list[str] = serving["generators"]
    classes = {g: cfg.models[g].get("model", g) for g in generators}
    hashes = {g: params_hash(cfg.models[g]) for g in generators}
    ds = RecDataset.load(processed_dir(split.dataset))
    from recsys.data.features import genre_matrix

    gm = genre_matrix(ds.items, split.item_ids)
    genres, genre_names = (gm[0], gm[1]) if gm else (None, [])
    years = ds.items.set_index("item_id")["year"].reindex(split.item_ids).astype("float64").to_numpy()
    n_cand = int(serving.get("candidates_per_generator", 40))

    def load(phase: str):
        missing = [g for g in generators if not model_path(split, g, hashes[g], phase).exists()]
        if missing:
            raise FileNotFoundError(
                f"saved models missing for {missing}: run `recsys benchmark -c {cfg.source.stem}`"
            )
        return scorers_from_models(
            {g: load_model(model_path(split, g, hashes[g], phase)) for g in generators}, classes
        )

    # 1) serving ranker: train-phase scorers on the train history, labels from validation
    train = split.train_data("train")
    seqs = sequences_of(train.interactions, split.n_users)
    scorers = load("train")
    _attach(scorers, train.matrix, seqs)
    ctx = Context(train.matrix, np.array([s[-1] if len(s) else -1 for s in seqs]), genres, years, None)
    val_targets = split.targets("val")
    val_users = np.flatnonzero(np.diff(val_targets.indptr) > 0)
    # Large datasets: train the ranker and evaluate on a fixed random sample of users, so the
    # (users x candidates) feature tables fit in memory. Settings: ranker_max_users, eval_max_users.
    rng = np.random.default_rng(int(serving.get("seed", 42)))
    val_users = _sample_users(val_users, serving.get("ranker_max_users"), rng)
    table = build_features(scorers, val_users, ctx, n_candidates=n_cand, targets=val_targets)
    booster, info = train_ranker(table, params=serving.get("ranker"), seed=int(serving.get("seed", 42)))
    info.pop("held_users")

    # 2) final scorers (fitted on train + validation) and one evaluation on the test set
    trainval = split.train_data("train", "val")
    seqs_tv = sequences_of(trainval.interactions, split.n_users)
    final = load("trainval")
    _attach(final, trainval.matrix, seqs_tv)
    ctx_tv = Context(
        trainval.matrix, np.array([s[-1] if len(s) else -1 for s in seqs_tv]), genres, years, None
    )
    test_targets = split.targets("test")
    test_users = np.flatnonzero(np.diff(test_targets.indptr) > 0)
    sampled = _sample_users(test_users, serving.get("eval_max_users"), rng)
    if sampled.size < test_users.size:  # evaluate only the sampled users: other rows get no targets
        keep = np.zeros(split.n_users, dtype=np.float32)
        keep[sampled] = 1.0
        test_targets = (sp.diags(keep) @ test_targets).tocsr()
        test_targets.eliminate_zeros()
        test_users = sampled
    test_table = build_features(final, test_users, ctx_tv, n_candidates=n_cand, targets=test_targets)
    S = score_matrix(test_table, booster, (split.n_users, split.n_items))
    item_features = genres
    test = evaluate(HybridScores(S), trainval.matrix, test_targets, ks=cfg.ks, item_features=item_features)
    log.info("serving hybrid test %s=%.4f", cfg.primary_metric, test.metrics[cfg.primary_metric])

    # 3) write the bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    pop = np.asarray(trainval.matrix.sum(axis=0)).ravel().astype(np.float32)
    nb_idx, nb_sim = item_neighbours(trainval.matrix)
    items = ds.items.set_index("item_id").reindex(split.item_ids).reset_index()
    catalog = pd.DataFrame(  # (not "table": that name holds the ranker's feature table)
        {
            "item_idx": np.arange(split.n_items, dtype=np.int32),
            "movie_id": split.item_ids.astype(np.int32),
            "title": items["title"].astype(str),
            "year": items["year"].astype("Int16") if "year" in items else pd.NA,
            "genres": items["genres"] if "genres" in items else [[] for _ in range(split.n_items)],
            "popularity": pop,
        }
    )
    # optional metadata from `recsys tmdb` (ids work without a key; posters/overviews need one)
    for extra, cols in (
        ("links.parquet", ["tmdb_id", "imdb_id"]),
        ("tmdb.parquet", ["poster_path", "overview", "tagline", "runtime", "cast"]),
    ):
        path = processed_dir(split.dataset) / extra
        if path.exists():
            meta = pd.read_parquet(path).drop_duplicates("item_id").set_index("item_id")
            catalog = catalog.join(meta[[c for c in cols if c in meta]], on="movie_id")
    catalog.to_parquet(out_dir / "items.parquet", index=False)
    np.savez_compressed(out_dir / "neighbours.npz", idx=nb_idx, sim=nb_sim.astype(np.float32))
    np.save(out_dir / "seeds.npy", onboarding_seeds(ds, split.item_ids))
    np.savez_compressed(  # dataset users' histories (train + validation), for the Model Lab
        out_dir / "dataset_users.npz",
        user_ids=split.user_ids.astype(np.int32),
        indptr=np.concatenate([[0], np.cumsum([len(s) for s in seqs_tv])]).astype(np.int64),
        items=np.concatenate(seqs_tv).astype(np.int32),
    )
    stem = cfg.source.stem if cfg.source else ""
    report_name = stem.removeprefix(split.dataset).strip("-") or "default_hparams"
    offline = results_docs_dir(split.dataset, report_name, "test.csv")  # the offline results table
    if offline.exists():
        (out_dir / "offline_results.csv").write_bytes(offline.read_bytes())
    if genres is not None:
        sp.save_npz(out_dir / "genres.npz", genres.tocsr())
    for name, scorer in final.items():
        _save_scorer(out_dir, name, scorer)
    booster.save_model(str(out_dir / "ranker.txt"), num_iteration=booster.best_iteration)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": split.dataset,
        "split": split.config.name,
        "split_fingerprint": split.stats["fingerprint"],
        "config": cfg.source.name if cfg.source else None,
        "generators": {g: {"model": classes[g], "settings_hash": hashes[g]} for g in generators},
        "candidates_per_generator": n_cand,
        "feature_names": table.names,
        "genre_names": genre_names,
        "n_items": split.n_items,
        "ranker": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()},
        "test_metrics": {k: float(v) for k, v in test.metrics.items()},
        "test_users_evaluated": int(test_users.size),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ExportReport(out_dir, manifest["test_metrics"], manifest["ranker"], time.perf_counter() - t0)


def _save_scorer(out_dir: Path, name: str, s: HistoryScorer) -> None:
    if isinstance(s, EASEScorer):
        np.save(out_dir / f"{name}.ease.npy", s.B.astype(np.float32))
    elif isinstance(s, IALSScorer):
        np.savez(out_dir / f"{name}.ials.npz", Q=s.Q.astype(np.float32), alpha=s.alpha, reg=s.reg)
    elif isinstance(s, MarkovScorer):
        sp.save_npz(out_dir / f"{name}.markov.npz", s.transitions)
        np.savez(out_dir / f"{name}.markov.meta.npz", popularity=s.popularity, pop_weight=s.pop_weight)
    elif isinstance(s, PopularityScorer):
        np.save(out_dir / f"{name}.mostpop.npy", s.popularity)
    elif isinstance(s, SASRecScorer):
        np.savez(out_dir / f"{name}.sasrec.npz", **s.w)
        cfg = {"dim": s.dim, "heads": s.heads, "blocks": s.blocks, "max_len": s.max_len}
        (out_dir / f"{name}.sasrec.json").write_text(json.dumps(cfg), encoding="utf-8")
    else:
        raise TypeError(type(s))


def load_scorers(bundle: Path, manifest: dict) -> dict[str, HistoryScorer]:
    """Rebuild the serving scorers from a bundle folder (NumPy + SciPy only)."""
    out: dict[str, HistoryScorer] = {}
    for name, spec in manifest["generators"].items():
        cls = spec["model"]
        if cls == "ease":
            out[name] = EASEScorer(np.load(bundle / f"{name}.ease.npy"))
        elif cls == "ials":
            z = np.load(bundle / f"{name}.ials.npz")
            out[name] = IALSScorer(z["Q"], float(z["alpha"]), float(z["reg"]))
        elif cls == "markov":
            meta = np.load(bundle / f"{name}.markov.meta.npz")
            out[name] = MarkovScorer(
                sp.load_npz(bundle / f"{name}.markov.npz"), meta["popularity"], float(meta["pop_weight"])
            )
        elif cls == "mostpop":
            out[name] = PopularityScorer(np.load(bundle / f"{name}.mostpop.npy"))
        elif cls == "sasrec":
            cfg = json.loads((bundle / f"{name}.sasrec.json").read_text(encoding="utf-8"))
            out[name] = SASRecScorer(dict(np.load(bundle / f"{name}.sasrec.npz")), cfg)
        else:
            raise ValueError(cls)
    return out


def load_ranker(bundle: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(bundle / "ranker.txt"))


def evaluate_serving(cfg: ExperimentConfig, split: Split, bundle_dir: Path) -> tuple[dict, pd.DataFrame]:
    """The served hybrid and each of its generators on the same sampled test users as `export_bundle`
    (the same seed gives the same sample), with per-user results, so the hybrid's gain can be tested for
    significance (Wilcoxon signed-rank with Holm correction). The bundle's trained ranker is used as it
    is: nothing is retrained. Returns ({model: EvalResult}, the paired tests)."""
    from recsys.analysis.results import paired_tests
    from recsys.data.features import genre_matrix

    serving = cfg.serving
    generators: list[str] = serving["generators"]
    classes = {g: cfg.models[g].get("model", g) for g in generators}
    hashes = {g: params_hash(cfg.models[g]) for g in generators}
    ds = RecDataset.load(processed_dir(split.dataset))
    gm = genre_matrix(ds.items, split.item_ids)
    genres = gm[0] if gm else None
    years = ds.items.set_index("item_id")["year"].reindex(split.item_ids).astype("float64").to_numpy()

    # the export drew the ranker's validation users first, then the test users: repeat both draws
    rng = np.random.default_rng(int(serving.get("seed", 42)))
    val_targets = split.targets("val")
    _sample_users(np.flatnonzero(np.diff(val_targets.indptr) > 0), serving.get("ranker_max_users"), rng)
    test_targets = split.targets("test")
    test_users = np.flatnonzero(np.diff(test_targets.indptr) > 0)
    sampled = _sample_users(test_users, serving.get("eval_max_users"), rng)
    if sampled.size < test_users.size:
        keep = np.zeros(split.n_users, dtype=np.float32)
        keep[sampled] = 1.0
        test_targets = (sp.diags(keep) @ test_targets).tocsr()
        test_targets.eliminate_zeros()

    trainval = split.train_data("train", "val")
    seqs_tv = sequences_of(trainval.interactions, split.n_users)
    models = {g: load_model(model_path(split, g, hashes[g], "trainval")) for g in generators}
    final = scorers_from_models(models, classes)
    _attach(final, trainval.matrix, seqs_tv)
    ctx = Context(trainval.matrix, np.array([s[-1] if len(s) else -1 for s in seqs_tv]), genres, years, None)
    table = build_features(
        final,
        sampled,
        ctx,
        n_candidates=int(serving.get("candidates_per_generator", 40)),
        targets=test_targets,
    )
    S = score_matrix(table, load_ranker(bundle_dir), (split.n_users, split.n_items))
    del table
    results = {
        "hybrid": evaluate(HybridScores(S), trainval.matrix, test_targets, ks=cfg.ks, item_features=genres)
    }
    log.info(
        "served hybrid on %d test users: ndcg@10 %.4f", sampled.size, results["hybrid"].metrics["ndcg@10"]
    )
    del S
    for g in generators:
        results[g] = evaluate(models[g], trainval.matrix, test_targets, ks=cfg.ks, item_features=genres)
        log.info("%s on the same users: ndcg@10 %.4f", g, results[g].metrics["ndcg@10"])
    per_user = pd.concat([r.per_user.assign(model=name) for name, r in results.items()], ignore_index=True)
    return results, paired_tests(per_user, "hybrid")
