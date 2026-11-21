"""Model registry. Models are imported lazily so non-PyTorch models never pay for importing torch."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from recsys.models.base import Recommender

_REGISTRY: dict[str, str] = {
    "mostpop": "recsys.models.popularity:MostPop",
    "userknn": "recsys.models.knn:UserKNN",
    "itemknn": "recsys.models.knn:ItemKNN",
    "ease": "recsys.models.ease:EASE",
    "bpr": "recsys.models.bpr:BPRMF",
    "ials": "recsys.models.ials:IALS",
    "neumf": "recsys.models.neumf:NeuMF",
    "multvae": "recsys.models.multvae:MultVAE",
    "lightgcn": "recsys.models.lightgcn:LightGCN",
    "sasrec": "recsys.models.sasrec:SASRec",
    "markov": "recsys.models.markov:MarkovChain",
}

MODEL_NAMES = tuple(_REGISTRY)


def get_model_class(name: str) -> type[Recommender]:
    try:
        module, cls = _REGISTRY[name].split(":")
    except KeyError:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(_REGISTRY)}") from None
    return getattr(import_module(module), cls)


def build(name: str, **params: Any) -> Recommender:
    return get_model_class(name)(**params)
