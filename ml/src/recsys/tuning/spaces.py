"""Search spaces: which hyperparameters are tuned per model, and over which ranges.

Ranges bracket the values recommended in each model's paper. Learning rates and regularization strengths
are searched on a log scale, because their effect is multiplicative. Structural choices with only a few
sensible values are categorical.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import optuna

Space = Callable[[optuna.Trial], dict[str, Any]]


def _knn(t: optuna.Trial) -> dict[str, Any]:
    return {"k": t.suggest_int("k", 10, 800, log=True), "shrink": t.suggest_float("shrink", 0.0, 200.0)}


SPACES: dict[str, Space] = {
    "userknn": _knn,
    "itemknn": _knn,
    "ease": lambda t: {"l2": t.suggest_float("l2", 10.0, 20000.0, log=True)},
    "bpr": lambda t: {
        "dim": t.suggest_categorical("dim", [32, 64, 128]),
        "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
        "reg": t.suggest_float("reg", 1e-6, 1e-1, log=True),
    },
    "ials": lambda t: {
        "dim": t.suggest_categorical("dim", [32, 64, 128, 256]),
        "alpha": t.suggest_float("alpha", 0.5, 100.0, log=True),
        "reg": t.suggest_float("reg", 1e-2, 1000.0, log=True),
    },
    "neumf": lambda t: {
        "gmf_dim": t.suggest_categorical("gmf_dim", [16, 32, 64]),
        # Optuna categorical choices must be primitives, so tower shapes are encoded as JSON strings
        "mlp_dims": json.loads(
            t.suggest_categorical("mlp_dims", ["[32, 16, 8]", "[64, 32, 16]", "[128, 64, 32]"])
        ),
        "negatives": t.suggest_categorical("negatives", [2, 4, 8]),
        "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
    },
    "multvae": lambda t: {
        "dropout": t.suggest_float("dropout", 0.1, 0.7),
        "anneal_cap": t.suggest_float("anneal_cap", 0.05, 1.0, log=True),
        "anneal_epochs": t.suggest_categorical("anneal_epochs", [20, 50, 100]),
        "lr": t.suggest_float("lr", 1e-4, 3e-3, log=True),
    },
    "lightgcn": lambda t: {
        "layers": t.suggest_int("layers", 1, 4),
        "lr": t.suggest_float("lr", 5e-4, 1e-2, log=True),
        "reg": t.suggest_float("reg", 1e-6, 1e-2, log=True),
    },
    "sasrec": lambda t: {
        "blocks": t.suggest_int("blocks", 1, 3),
        "dropout": t.suggest_float("dropout", 0.1, 0.6),
        "lr": t.suggest_float("lr", 2e-4, 3e-3, log=True),
        "max_len": t.suggest_categorical("max_len", [50, 100, 200]),
    },
}


def has_space(model_class: str) -> bool:
    return model_class in SPACES
