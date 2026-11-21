"""A tiny but complete serving bundle for tests (also used by the backend's test suite)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.split import TrainData
from recsys.hybrid.features import Context, build_features
from recsys.hybrid.ranker import train_ranker
from recsys.models.ease import EASE
from recsys.models.markov import MarkovChain
from recsys.models.popularity import MostPop
from recsys.serving.bundle import _save_scorer, item_neighbours, sequences_of
from recsys.serving.scorers import EASEScorer, MarkovScorer, PopularityScorer

GENRES = ["Action", "Comedy", "Drama"]


def block_matrix(n_users: int = 40, n_items: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two taste groups: first half of the users likes the first half of the items (7 of 10 seen)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_users, n_items))
    held = np.zeros_like(X, dtype=bool)
    for u in range(n_users):
        group = np.arange(n_items // 2) if u < n_users // 2 else np.arange(n_items // 2, n_items)
        seen = rng.choice(group, size=7, replace=False)
        X[u, seen] = 1
        held[u, np.setdiff1d(group, seen)] = True
    return X, held


class KeywordEncoder:
    """Stand-in for the ONNX sentence encoder in tests: a text mentioning a genre points along that
    genre's axis (the tiny bundle's movie vectors are their genre axes)."""

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), len(GENRES)), dtype=np.float32)
        for r, text in enumerate(texts):
            for j, g in enumerate(GENRES):
                out[r, j] = float(g.lower() in text.lower())
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def make_tiny_bundle(path: Path, with_forecast: bool = False) -> Path:
    """A complete serving bundle over 40 viewers x 20 films; with_forecast adds the crowd forecast
    (trending and cold start by genre) built from 20 of the viewers."""
    X, held = block_matrix()
    M = sp.csr_matrix(X.astype(np.float32))
    u, i = M.nonzero()
    data = TrainData(
        M, pd.DataFrame({"user_idx": u, "item_idx": i, "rating": 5.0, "timestamp": np.arange(u.size)})
    )
    seqs = sequences_of(data.interactions, X.shape[0])
    n_items = X.shape[1]
    genres = sp.csr_matrix(np.eye(len(GENRES), dtype=np.float32)[np.arange(n_items) % len(GENRES)])
    mc = MarkovChain().fit(data)
    scorers = {
        "ease": EASEScorer(EASE(l2=5.0).fit(data).B_),
        "markov": MarkovScorer(mc.transitions_, mc.popularity_, 1e-6),
        "mostpop": PopularityScorer(MostPop().fit(data).popularity_),
    }
    for s in scorers.values():
        s.set_history(data.matrix, seqs)
    ctx = Context(data.matrix, np.array([s[-1] for s in seqs]), genres, np.arange(n_items, dtype=float), None)
    targets = sp.csr_matrix(held.astype(np.float32))
    table = build_features(scorers, np.arange(X.shape[0]), ctx, n_candidates=6, targets=targets)
    booster, _ = train_ranker(table, rounds=30, early_stop=10, seed=0)

    path.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path / "ranker.txt"))
    for name, s in scorers.items():
        _save_scorer(path, name, s)
    pd.DataFrame(
        {
            "item_idx": np.arange(n_items, dtype=np.int32),
            "movie_id": np.arange(100, 100 + n_items, dtype=np.int32),
            "title": [f"Movie {i}" for i in range(n_items)],
            "year": pd.array(np.arange(1990, 1990 + n_items), dtype="Int16"),
            "genres": [[GENRES[i % len(GENRES)]] for i in range(n_items)],
            "popularity": np.asarray(M.sum(axis=0)).ravel().astype(np.float32),
            "keywords": [[GENRES[i % len(GENRES)].lower(), f"tag{i}"] for i in range(n_items)],
        }
    ).to_parquet(path / "items.parquet", index=False)
    np.save(path / "text_embeddings.npy", genres.toarray().astype(np.float32))  # content vectors
    idx, sim = item_neighbours(M, k=5)
    np.savez_compressed(path / "neighbours.npz", idx=idx, sim=sim)
    np.save(path / "seeds.npy", np.arange(8, dtype=np.int32))
    sp.save_npz(path / "genres.npz", genres)
    np.savez_compressed(  # MovieLens users' histories, for the Model Lab
        path / "dataset_users.npz",
        user_ids=np.arange(1, X.shape[0] + 1, dtype=np.int32),
        indptr=np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64),
        items=np.concatenate(seqs).astype(np.int32),
    )
    manifest = {
        "bundle_version": 1,
        "dataset": "tiny",
        "created": "test",
        "generators": {n: {"model": n} for n in scorers},
        "candidates_per_generator": 6,
        "feature_names": table.names,
        "genre_names": GENRES,
        "n_items": n_items,
        "test_metrics": {"ndcg@10": 0.5},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if with_forecast:
        from recsys.serving.forecast import add_forecast_to_bundle  # needs the finished bundle

        add_forecast_to_bundle(path, n_viewers=20, seed=0)
    return path
