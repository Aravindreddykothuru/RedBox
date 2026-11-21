"""The interface every recommender implements.

Keeping one small interface means the evaluator, experiment runner, and later the web backend never
need to know which algorithm they are talking to.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp

from recsys.data.split import TrainData

log = logging.getLogger("recsys.models")

# Called by iterative models during training; returns the validation score (higher is better).
Validator = Callable[["Recommender"], float]


class Recommender(ABC):
    name: ClassVar[str]
    defaults: ClassVar[dict[str, Any]] = {}
    iterative: ClassVar[bool] = False  # True if fit() uses a validator for early stopping

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown parameters {sorted(unknown)}")
        self.params: dict[str, Any] = {**self.defaults, **params}

    @abstractmethod
    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        """Learn from ``data``. Iterative models may call ``validator`` for early stopping."""

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Scores for every item, shape (len(users), n_items). Higher means more recommended."""

    def recommend(self, users: np.ndarray, k: int, exclude: sp.csr_matrix | None = None) -> np.ndarray:
        """Top-k item indices per user, best first, skipping items in ``exclude`` (e.g. history)."""
        from recsys.eval.evaluator import _fill, top_k

        users = np.asarray(users)
        scores = np.array(self.score(users), dtype=np.float32, copy=True)
        if exclude is not None:
            _fill(scores, exclude.tocsr()[users], -np.inf)
        return top_k(scores, k)

    def refit_params(self) -> dict[str, Any]:
        """Parameters to freeze when refitting on train+validation (e.g. the best epoch count)."""
        return {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


class IterativeRecommender(Recommender):
    """A model trained epoch by epoch, with early stopping on a validation score.

    Subclasses implement ``_run_epoch`` (one pass over the data, returns the mean loss),
    ``_snapshot`` (copy of the weights) and ``_restore``. ``_train`` runs the shared loop: every
    ``eval_every`` epochs the validator is called, the best weights are kept, and training stops after
    ``patience`` evaluations without improvement. Without a validator (the refit on train+validation)
    it trains for exactly ``epochs`` epochs, i.e. the best epoch count found before.
    Subclass ``defaults`` must include ``epochs``, ``eval_every`` and ``patience``.
    """

    iterative = True

    @abstractmethod
    def _run_epoch(self, epoch: int) -> float: ...

    @abstractmethod
    def _snapshot(self) -> Any: ...

    @abstractmethod
    def _restore(self, state: Any) -> None: ...

    def _train(self, validator: Validator | None) -> None:
        p = self.params
        self.history_: list[dict] = []
        best_score, best_epoch, best_state, bad_evals = -np.inf, 0, None, 0
        epoch = 0
        for epoch in range(1, p["epochs"] + 1):
            record: dict = {"epoch": epoch, "loss": float(self._run_epoch(epoch))}
            if validator is not None and epoch % p["eval_every"] == 0:
                record["val"] = val = validator(self)
                log.info("%s epoch %3d  loss %.4f  val %.4f", self.name, epoch, record["loss"], val)
                if val > best_score:
                    best_score, best_epoch, bad_evals = val, epoch, 0
                    best_state = self._snapshot()
                else:
                    bad_evals += 1
            self.history_.append(record)
            if validator is not None and bad_evals >= p["patience"]:
                log.info(
                    "%s early stop at epoch %d (best %d, val %.4f)", self.name, epoch, best_epoch, best_score
                )
                break
        if best_state is not None:
            self._restore(best_state)
            self.best_epoch_ = best_epoch
        else:
            self.best_epoch_ = epoch

    def refit_params(self) -> dict[str, Any]:
        return {"epochs": self.best_epoch_}
