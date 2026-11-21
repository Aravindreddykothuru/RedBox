"""Rating prediction (experiment E7): predict the star rating a user would give a movie.

Top-K ranking is the main task of this project. Rating prediction, the task of the Netflix Prize
(error measured by RMSE), is reported for completeness and for comparison with older literature.
All ratings count here (no "liked" threshold), with the same per-user chronological split and the same
protocol: fit on train -> early-stop on validation -> refit on train+validation -> test once.

Models
  GlobalMean   r = mu
  Biases       r = mu + b_u + b_i                 (regularized averages, Koren 2010)
  BiasedMF     r = mu + b_u + b_i + p_u . q_i     ("SVD" of the Netflix Prize; Koren, Bell, Volinsky 2009)
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from recsys.data.split import Split

log = logging.getLogger(__name__)
RATING_RANGE = (1.0, 5.0)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # copy=True: pandas 3 returns read-only views, which torch.from_numpy warns about
    return (
        df["user_idx"].to_numpy(np.int64, copy=True),
        df["item_idx"].to_numpy(np.int64, copy=True),
        df["rating"].to_numpy(np.float64, copy=True),
    )


class GlobalMean:
    name = "Global mean"

    def __init__(self, **_: Any) -> None:
        self.params: dict[str, Any] = {}

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, val: pd.DataFrame | None = None):
        self.mu_ = float(train["rating"].mean())
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.full(len(users), self.mu_)


class Biases:
    """mu + b_u + b_i with shrinkage: an item rated by few users gets a bias pulled towards 0."""

    name = "User + item biases"

    def __init__(self, reg_user: float = 10.0, reg_item: float = 25.0, iterations: int = 10):
        self.params = {"reg_user": reg_user, "reg_item": reg_item, "iterations": iterations}

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, val: pd.DataFrame | None = None):
        u, i, r = _arrays(train)
        self.mu_ = float(r.mean())
        self.bu_, self.bi_ = np.zeros(n_users), np.zeros(n_items)
        n_u = np.bincount(u, minlength=n_users)
        n_i = np.bincount(i, minlength=n_items)
        for _ in range(self.params["iterations"]):
            self.bi_ = np.bincount(i, r - self.mu_ - self.bu_[u], n_items) / (self.params["reg_item"] + n_i)
            self.bu_ = np.bincount(u, r - self.mu_ - self.bi_[i], n_users) / (self.params["reg_user"] + n_u)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.clip(self.mu_ + self.bu_[users] + self.bi_[items], *RATING_RANGE)


class BiasedMF:
    """mu + b_u + b_i + p_u . q_i, trained on squared error with L2, Adam, early stopping on val RMSE."""

    name = "Biased MF (SVD)"

    def __init__(self, dim=64, lr=1e-3, reg=0.02, batch_size=8192, epochs=100, patience=3, seed=42):
        self.params = dict(
            dim=dim, lr=lr, reg=reg, batch_size=batch_size, epochs=epochs, patience=patience, seed=seed
        )

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, val: pd.DataFrame | None = None):
        p = self.params
        torch.manual_seed(p["seed"])
        rng = np.random.default_rng(p["seed"])
        u, i, r = _arrays(train)
        self.mu_ = float(r.mean())
        self.P = torch.nn.Parameter(torch.randn(n_users, p["dim"]) * 0.01)
        self.Q = torch.nn.Parameter(torch.randn(n_items, p["dim"]) * 0.01)
        self.bu = torch.nn.Parameter(torch.zeros(n_users))
        self.bi = torch.nn.Parameter(torch.zeros(n_items))
        opt = torch.optim.Adam([self.P, self.Q, self.bu, self.bi], lr=p["lr"])
        ut, it, rt = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(r).float()

        best, best_epoch, best_state, bad = np.inf, 0, None, 0
        self.history_: list[dict] = []
        epoch = 0
        for epoch in range(1, p["epochs"] + 1):
            order = torch.from_numpy(rng.permutation(len(r)))
            for s in range(0, len(r), p["batch_size"]):
                b = order[s : s + p["batch_size"]]
                bu, bi, pu, qi = self.bu[ut[b]], self.bi[it[b]], self.P[ut[b]], self.Q[it[b]]
                pred = self.mu_ + bu + bi + (pu * qi).sum(1)
                l2 = (pu.pow(2).sum() + qi.pow(2).sum() + bu.pow(2).sum() + bi.pow(2).sum()) / b.numel()
                loss = torch.mean((rt[b] - pred) ** 2) + p["reg"] * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
            record = {"epoch": epoch}
            if val is not None:
                vu, vi, vr = _arrays(val)
                record["val_rmse"] = score = rmse(vr, self.predict(vu, vi))
                if score < best:
                    best, best_epoch, bad = score, epoch, 0
                    best_state = [t.detach().clone() for t in (self.P, self.Q, self.bu, self.bi)]
                else:
                    bad += 1
            self.history_.append(record)
            if val is not None and bad >= p["patience"]:
                break
        if best_state is not None:
            with torch.no_grad():
                for t, saved in zip((self.P, self.Q, self.bu, self.bi), best_state, strict=True):
                    t.copy_(saved)
        self.best_epoch_ = best_epoch if best_state is not None else epoch
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            u, i = torch.as_tensor(users), torch.as_tensor(items)
            pred = self.mu_ + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        return np.clip(pred.numpy().astype(np.float64), *RATING_RANGE)


@dataclass
class RatingResult:
    model: str
    params: dict[str, Any]
    val_rmse: float
    val_mae: float
    test_rmse: float
    test_mae: float
    fit_seconds: float


def run_rating_benchmark(split: Split) -> list[RatingResult]:
    """Evaluate every rating model with the train -> val -> refit -> test protocol."""
    if split.config.positive_threshold is not None:
        raise ValueError("rating prediction needs a split that keeps every rating (positive_threshold=None)")
    trainval = pd.concat([split.train, split.val], ignore_index=True)
    vu, vi, vr = _arrays(split.val)
    tu, ti, tr = _arrays(split.test)
    results = []
    for model in (GlobalMean(), Biases(), BiasedMF()):
        t0 = time.perf_counter()
        model.fit(split.train, split.n_users, split.n_items, val=split.val)
        fit_seconds = time.perf_counter() - t0
        val_pred = model.predict(vu, vi)
        params = dict(model.params)
        if hasattr(model, "best_epoch_"):
            params["epochs"] = model.best_epoch_  # refit for exactly the best number of epochs
        final = type(model)(**params)
        final.fit(trainval, split.n_users, split.n_items)
        test_pred = final.predict(tu, ti)
        results.append(
            RatingResult(model.name, params, rmse(vr, val_pred), mae(vr, val_pred),
                         rmse(tr, test_pred), mae(tr, test_pred), fit_seconds)
        )  # fmt: skip
        log.info(
            "%-22s val RMSE %.4f  test RMSE %.4f", model.name, results[-1].val_rmse, results[-1].test_rmse
        )
    return results


def all_ratings_config(split_config):
    """The same split settings, but keeping every rating."""
    return dataclasses.replace(split_config, positive_threshold=None)


def results_markdown(results: list[RatingResult]) -> str:
    best = min(r.test_rmse for r in results)
    lines = [
        "| Model | Val RMSE | Val MAE | Test RMSE | Test MAE | Fit (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rm = f"**{r.test_rmse:.4f}**" if r.test_rmse == best else f"{r.test_rmse:.4f}"
        cells = [
            r.model,
            f"{r.val_rmse:.4f}",
            f"{r.val_mae:.4f}",
            rm,
            f"{r.test_mae:.4f}",
            f"{r.fit_seconds:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
