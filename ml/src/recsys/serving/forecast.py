"""Crowd forecast: what the hybrid predicts a sample of the dataset's viewers will watch next, summed into
one score per film. It replaces plain popularity in the two places that had no personal history to use:

- the "to watch next" row (Engine.trending): films that many viewers are *predicted* to watch next, rather
  than films most viewers have already watched;
- brand-new users (Engine.cold_start): the same forecast, with each viewer weighted by how well their genre
  taste matches the genres the new user chose ("viewers like you"), or unweighted if they chose none.

Each viewer's top-10 counts with the NDCG discount, 1/log2(rank + 1), so a first place weighs most. The
bundle keeps only these per-film scores (forecast.npz), no viewer's history, so a public deployment can
ship it. `evaluate_forecast` checks both uses on the test split, against popularity, with the forecast built
from one group of viewers and judged on another (`recsys forecast`, docs/results/<dataset>/forecast/).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.serving.engine import Engine

K = 10
RANK_WEIGHT = 1 / np.log2(np.arange(2, K + 2))


def _viewers(engine: Engine, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A seeded split of the dataset's viewers: the first n build the forecast, the rest can judge it."""
    users = engine._dataset_users
    if users is None:
        raise ValueError(f"{engine.bundle} has no dataset_users.npz: the forecast needs viewers' histories")
    order = np.random.default_rng(seed).permutation(len(users["user_ids"]))
    return order[:n], order[n:]


def _history(engine: Engine, pos: int) -> np.ndarray:
    users = engine._dataset_users
    return users["items"][users["indptr"][pos] : users["indptr"][pos + 1]]


def _taste(engine: Engine, history: np.ndarray) -> np.ndarray | None:
    """A viewer's genre taste: the genres of the films they liked, as a unit vector."""
    if engine.genres_dense is None or not len(history):
        return None
    p = np.asarray(engine.genres_dense[history].sum(0), dtype=np.float64).ravel()
    norm = np.linalg.norm(p)
    return p / norm if norm else None


def build_forecast(engine: Engine, viewers: np.ndarray) -> dict[str, np.ndarray]:
    """Run the hybrid for every viewer and add up their top-10s: overall and per genre of taste."""
    n_genres = 0 if engine.genres_dense is None else engine.genres_dense.shape[1]
    crowd = np.zeros(engine.n_items)
    by_genre = np.zeros((n_genres, engine.n_items))
    used = 0
    for pos in viewers:
        history = _history(engine, int(pos))
        if not len(history):
            continue
        top = np.array([r.item_idx for r in engine.recommend(history.tolist(), k=K)], dtype=np.int64)
        weights = RANK_WEIGHT[: len(top)]
        crowd[top] += weights
        taste = _taste(engine, history)
        if taste is not None:
            by_genre[:, top] += np.outer(taste, weights)
        used += 1
    return {
        "crowd": crowd.astype(np.float32),
        "by_genre": by_genre.astype(np.float32),
        "n_viewers": np.array(used, dtype=np.int32),
    }


def evaluate_forecast(engine: Engine, forecast: dict, test_targets, judges: np.ndarray) -> pd.DataFrame:
    """How well each way of choosing films predicts what viewers liked next (the test split; the models
    never saw it), compared with plain popularity.

    - trending: rank all films by each score and compare with the films viewers actually liked next
      (NDCG@10 and NDCG@50 with the number of such viewers as the gain, and the overlap of the two top 20s);
    - cold start: for each judging viewer (none of them helped build the forecast), pretend they are new
      and chose their 3 most-liked genres; hit rate and NDCG of 10 films against what they liked next."""
    targets = test_targets.tocsr()
    demand = np.asarray(targets.sum(0), dtype=np.float64).ravel()
    pop = engine.popularity.astype(np.float64)
    crowd = forecast["crowd"].astype(np.float64)
    discount = 1 / np.log2(np.arange(2, 52))

    def ndcg(score: np.ndarray, k: int) -> float:
        k = min(k, len(demand))  # small catalogues (tests)
        top = np.argsort(-score, kind="stable")[:k]
        return float((demand[top] * discount[:k]).sum() / (np.sort(demand)[::-1][:k] * discount[:k]).sum())

    def overlap(score: np.ndarray, k: int = 20) -> float:
        return len(set(np.argsort(-score)[:k]) & set(np.argsort(-demand)[:k])) / k

    rows = [
        {
            "use": "trending",
            "method": name,
            "NDCG@10": ndcg(s, 10),
            "NDCG@50": ndcg(s, 50),
            "top-20 overlap": overlap(s),
        }
        for name, s in (("popularity", pop), ("ML crowd forecast", crowd))
    ]

    genres = engine.genres_dense
    results: dict[str, list[tuple[float, float]]] = {}
    for pos in judges:
        target = targets.indices[targets.indptr[pos] : targets.indptr[pos + 1]]
        history = _history(engine, int(pos))
        if not len(target) or not len(history) or genres is None:
            continue
        chosen = np.argsort(-np.asarray(genres[history].sum(0)).ravel())[:3]
        in_chosen = np.asarray(genres[:, chosen].sum(1)).ravel() > 0
        methods = {
            "popularity": pop,
            "popular in the chosen genres": pop * in_chosen,
            "ML crowd forecast (no genres)": crowd,
            "ML viewers like you (chosen genres)": forecast["by_genre"][chosen].sum(0).astype(np.float64),
        }
        liked = set(target.tolist())
        ideal = (1 / np.log2(np.arange(2, min(K, len(liked)) + 2))).sum()
        for name, s in methods.items():
            hits = np.array([i in liked for i in np.argsort(-s, kind="stable")[:K]], dtype=np.float64)
            results.setdefault(name, []).append(
                (float(hits.any()), float((hits * RANK_WEIGHT).sum() / ideal))
            )
    for name, scores in results.items():
        hit, nd = np.mean(scores, axis=0)
        rows.append(
            {"use": "cold start", "method": name, "hit@10": hit, "NDCG@10": nd, "viewers": len(scores)}
        )
    return pd.DataFrame(rows)


def add_forecast_to_bundle(bundle: Path, n_viewers: int = 4000, seed: int = 42) -> tuple[dict, np.ndarray]:
    """Build the forecast for a bundle and save it as forecast.npz (+ a note in the manifest). Returns the
    forecast and the viewers not used to build it, for evaluate_forecast."""
    bundle = Path(bundle)
    engine = Engine(bundle)
    build, judges = _viewers(engine, n_viewers, seed)
    t0 = time.perf_counter()
    forecast = build_forecast(engine, build)
    np.savez_compressed(bundle / "forecast.npz", **forecast)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["forecast"] = {"viewers": int(forecast["n_viewers"]), "top_k": K, "seed": seed}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    forecast["seconds"] = np.array(time.perf_counter() - t0)
    return forecast, judges
