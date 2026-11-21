"""MostPop: recommend the most-interacted items to everyone. The non-personalized floor to beat."""

from __future__ import annotations

from typing import Self

import numpy as np

from recsys.data.split import TrainData
from recsys.models.base import Recommender, Validator


class MostPop(Recommender):
    name = "mostpop"

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        self.popularity_ = np.asarray(data.matrix.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.popularity_, (len(users), self.popularity_.size))
