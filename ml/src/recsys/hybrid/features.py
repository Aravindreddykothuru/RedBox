"""Stage 1 (candidate generation) and the feature table for stage 2 (the learned ranker).

For every user, each generator model proposes its top-n items; the union is the candidate set. Every
(user, candidate) pair gets one feature row. Generator signals are made comparable across users by
standardizing scores within the user's row (z-score) and by using the item's rank in the full catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.eval.evaluator import _fill
from recsys.models.base import Recommender


@dataclass
class Context:
    """Everything known at prediction time that is not a generator: history and side information."""

    history: sp.csr_matrix  # (users, items) known interactions; never candidates
    last_item: np.ndarray  # (users,) most recent history item, -1 if none
    item_genres: sp.csr_matrix | None  # (items, genres) multi-hot
    item_year: np.ndarray | None  # (items,) release year (nan if unknown)
    user_side: pd.DataFrame | None  # optional demographic columns, indexed by user_idx
    # Global item statistics. Offline they are computed from ``history`` (all users). When serving one
    # user, ``history`` is a single row, so they MUST be passed in (computed from the training data),
    # otherwise popularity / similarity features would describe one user instead of everyone.
    item_pop: np.ndarray | None = None  # (items,) number of users who liked each item
    item_sim: np.ndarray | None = None  # (items, items) cosine similarity


@dataclass
class FeatureTable:
    users: np.ndarray  # (rows,) user_idx of each pair
    items: np.ndarray  # (rows,) item_idx of each pair
    X: np.ndarray  # (rows, features) float32
    names: list[str]
    labels: np.ndarray | None  # (rows,) 1 if the pair is a held-out target
    candidate_recall: float | None  # share of targets that made it into the candidate set


def last_items(log: pd.DataFrame, n_users: int) -> np.ndarray:
    last = log.groupby("user_idx")["item_idx"].last()
    out = np.full(n_users, -1, dtype=np.int64)
    out[last.index.to_numpy()] = last.to_numpy()
    return out


def item_cosine(history: sp.csr_matrix) -> np.ndarray:
    """Dense item-item cosine similarity on the binary history (items x items, float32)."""
    X = history.astype(np.float32)
    co = (X.T @ X).toarray()
    norms = np.sqrt(np.diag(co))
    return (co / (np.outer(norms, norms) + 1e-9)).astype(np.float32)


def build_features(
    generators: dict[str, Recommender],
    users: np.ndarray,
    ctx: Context,
    *,
    n_candidates: int,
    targets: sp.csr_matrix | None = None,
    batch_size: int = 512,
    extra_items: np.ndarray | None = None,
) -> FeatureTable:
    """Candidates and ranker features. ``extra_items`` are scored for every user as well, even when no
    generator proposed them (watch-together pools a group's candidates); history items never are."""
    history = ctx.history.tocsr()
    n_items = history.shape[1]
    item_pop = ctx.item_pop if ctx.item_pop is not None else np.asarray(history.sum(axis=0)).ravel()
    log_pop = np.log1p(item_pop).astype(np.float32)
    cosine = ctx.item_sim if ctx.item_sim is not None else item_cosine(history)
    genres = ctx.item_genres.toarray().astype(np.float32) if ctx.item_genres is not None else None
    genre_norm = np.linalg.norm(genres, axis=1) + 1e-9 if genres is not None else None
    n_hist_all = np.diff(history.indptr)

    names = []
    for g in generators:
        names += [f"{g}_z", f"{g}_logrank", f"{g}_in_top"]
    names += ["n_generators", "item_log_pop", "item_pop_vs_user", "cos_to_last"]
    if ctx.item_year is not None:
        names.append("item_year")
    if genres is not None:
        names += ["genre_affinity"] + [f"genre_{j}" for j in range(genres.shape[1])]
    names += ["user_log_history", "user_mean_log_pop"]
    if ctx.user_side is not None:
        names += [f"user_{c}" for c in ctx.user_side.columns]

    out_users, out_items, out_X, out_y = [], [], [], []
    n_targets_total = n_found = 0
    for s in range(0, users.size, batch_size):
        b = users[s : s + batch_size]
        hist_rows = history[b]
        z_cols, rank_cols, top_masks = [], [], []
        for model in generators.values():
            scores = np.array(model.score(b), dtype=np.float32, copy=True)
            _fill(scores, hist_rows, np.nan)  # history items are never candidates
            mean = np.nanmean(scores, axis=1, keepdims=True)
            std = np.nanstd(scores, axis=1, keepdims=True) + 1e-9
            z = (scores - mean) / std
            filled = np.where(np.isnan(scores), -np.inf, scores)
            order = np.argsort(-filled, axis=1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.arange(1, n_items + 1)[None, :].repeat(b.size, 0), axis=1)
            top = ranks <= n_candidates
            top &= ~np.isnan(scores)
            z_cols.append(z)
            rank_cols.append(ranks)
            top_masks.append(top)
        cand = np.logical_or.reduce(top_masks)
        if extra_items is not None and len(extra_items):
            cand[:, extra_items] = True
            seen = np.zeros_like(cand)
            _fill(seen, hist_rows, True)
            cand &= ~seen
        rows, cols = np.nonzero(cand)

        feats = []
        for z, ranks, top in zip(z_cols, rank_cols, top_masks, strict=True):
            feats += [
                z[rows, cols],
                np.log(ranks[rows, cols]).astype(np.float32),
                top[rows, cols].astype(np.float32),
            ]
        feats.append(np.sum([t[rows, cols] for t in top_masks], axis=0).astype(np.float32))
        u_global = b[rows]
        user_mean_pop = _row_mean(hist_rows, log_pop)
        feats += [log_pop[cols], log_pop[cols] - user_mean_pop[rows]]
        last = ctx.last_item[u_global]
        feats.append(np.where(last >= 0, cosine[np.maximum(last, 0), cols], 0.0).astype(np.float32))
        if ctx.item_year is not None:
            feats.append(ctx.item_year[cols].astype(np.float32))
        if genres is not None:
            profile = np.asarray(hist_rows @ genres)  # (batch, genres): genre counts in the history
            profile_norm = np.linalg.norm(profile, axis=1) + 1e-9
            affinity = (profile[rows] * genres[cols]).sum(axis=1) / (profile_norm[rows] * genre_norm[cols])
            feats.append(affinity.astype(np.float32))
            feats += list(genres[cols].T)
        feats += [np.log1p(n_hist_all[u_global]).astype(np.float32), user_mean_pop[rows]]
        if ctx.user_side is not None:
            side = ctx.user_side.reindex(u_global).to_numpy(dtype=np.float32)
            feats += list(side.T)

        out_users.append(u_global)
        out_items.append(cols)
        out_X.append(np.column_stack(feats).astype(np.float32))
        if targets is not None:
            t = targets[b].tocsr()
            relevant = np.zeros((b.size, n_items), dtype=bool)
            _fill(relevant, t, True)
            out_y.append(relevant[rows, cols].astype(np.float32))
            n_targets_total += t.nnz
            n_found += int(relevant[cand].sum())

    return FeatureTable(
        users=np.concatenate(out_users),
        items=np.concatenate(out_items),
        X=np.concatenate(out_X),
        names=names,
        labels=np.concatenate(out_y) if targets is not None else None,
        candidate_recall=(n_found / n_targets_total) if targets is not None and n_targets_total else None,
    )


def _row_mean(rows: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    counts = np.diff(rows.indptr)
    sums = np.asarray(rows @ values).ravel()
    return (sums / np.maximum(counts, 1)).astype(np.float32)
