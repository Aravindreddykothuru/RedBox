"""History-based scorers in plain NumPy (no PyTorch), one per serving generator.

Each scorer follows the Recommender interface: ``set_history`` gives it the users' interaction matrix
and chronological sequences, then ``score(users)`` scores every item for those users. The same objects
build the serving ranker's training features and answer live requests, so training and serving
features are computed by identical code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import Recommender


class HistoryScorer(Recommender):
    """Base class: scores come from the given history, not from user ids seen in training."""

    name = "history"

    def __init__(self) -> None:
        super().__init__()
        self.history: sp.csr_matrix | None = None
        self.sequences: list[np.ndarray] = []

    def set_history(self, history: sp.csr_matrix, sequences: list[np.ndarray]) -> HistoryScorer:
        self.history = history.tocsr()
        self.sequences = sequences
        return self

    def fit(self, data, validator=None):  # weights come from the exported bundle
        return self


class EASEScorer(HistoryScorer):
    name = "ease"

    def __init__(self, B: np.ndarray):
        super().__init__()
        self.B = B.astype(np.float32)

    def score(self, users):
        return np.asarray(self.history[np.asarray(users)] @ self.B)


class PopularityScorer(HistoryScorer):
    name = "mostpop"

    def __init__(self, popularity: np.ndarray):
        super().__init__()
        self.popularity = popularity.astype(np.float32)

    def score(self, users):
        return np.tile(self.popularity, (len(users), 1))


class MarkovScorer(HistoryScorer):
    """P(next | last item) from the exported transition matrix, plus a tiny popularity tie-breaker."""

    name = "markov"

    def __init__(self, transitions: sp.csr_matrix, popularity: np.ndarray, pop_weight: float = 1e-6):
        super().__init__()
        self.transitions = transitions.tocsr().astype(np.float32)
        self.popularity = popularity.astype(np.float32)
        self.pop_weight = pop_weight

    def score(self, users):
        users = np.asarray(users)
        out = np.tile(self.popularity * self.pop_weight, (users.size, 1))
        for r, u in enumerate(users):
            seq = self.sequences[u]
            if len(seq):
                out[r] += self.transitions[int(seq[-1])].toarray().ravel()
        return out


class IALSScorer(HistoryScorer):
    """iALS "fold-in": the exact closed-form user vector given the item factors (one ALS half-step),
    so a new user is placed in the model's latent space instantly."""

    name = "ials"

    def __init__(self, item_factors: np.ndarray, alpha: float, reg: float):
        super().__init__()
        self.Q = item_factors.astype(np.float64)
        self.alpha, self.reg = alpha, reg
        self.gram = self.Q.T @ self.Q + reg * np.eye(self.Q.shape[1])

    def user_vectors(self, users) -> np.ndarray:
        H = self.history[np.asarray(users)].tocsr()
        out = np.zeros((H.shape[0], self.Q.shape[1]))
        for r in range(H.shape[0]):
            idx = H.indices[H.indptr[r] : H.indptr[r + 1]]
            if idx.size:
                Qu = self.Q[idx]
                A = self.gram + self.alpha * (Qu.T @ Qu)
                out[r] = np.linalg.solve(A, (1.0 + self.alpha) * Qu.sum(axis=0))
        return out

    def score(self, users):
        return (self.user_vectors(users) @ self.Q.T).astype(np.float32)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class SASRecScorer(HistoryScorer):
    """SASRec inference re-implemented in NumPy from the trained PyTorch weights.

    It mirrors ``SASRecNet.forward`` in eval mode (no dropout), including padding masks and positions
    counted from the most recent item; a unit test checks it matches PyTorch to float precision."""

    name = "sasrec"

    def __init__(self, weights: dict[str, np.ndarray], config: dict[str, Any]):
        super().__init__()
        self.w = {k: v.astype(np.float32) for k, v in weights.items()}
        self.dim, self.heads = int(config["dim"]), int(config["heads"])
        self.blocks, self.max_len = int(config["blocks"]), int(config["max_len"])

    @classmethod
    def from_torch(cls, net, config: dict[str, Any]) -> SASRecScorer:
        weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return cls(weights, config)

    def hidden(self, seq: np.ndarray) -> np.ndarray:
        """seq: (batch, L) item ids + 1, left-padded with 0. Returns hidden states (batch, L, dim)."""
        w, d, h = self.w, self.dim, self.heads
        batch, length = seq.shape
        pad = seq == 0
        keep = (~pad)[..., None]
        x = w["item_emb.weight"][seq] * np.sqrt(d) + w["pos_emb.weight"][self.max_len - length :]
        x = x * keep
        causal = np.triu(np.ones((length, length), dtype=bool), k=1)
        blocked = causal[None] | (pad[:, None, :] & ~np.eye(length, dtype=bool)[None])
        hd = d // h
        for i in range(self.blocks):
            q_in = _layer_norm(x, w[f"attn_norms.{i}.weight"], w[f"attn_norms.{i}.bias"])
            W, bias = w[f"attns.{i}.in_proj_weight"], w[f"attns.{i}.in_proj_bias"]
            q = q_in @ W[:d].T + bias[:d]
            k = x @ W[d : 2 * d].T + bias[d : 2 * d]
            v = x @ W[2 * d :].T + bias[2 * d :]
            q = q.reshape(batch, length, h, hd).transpose(0, 2, 1, 3)
            k = k.reshape(batch, length, h, hd).transpose(0, 2, 1, 3)
            v = v.reshape(batch, length, h, hd).transpose(0, 2, 1, 3)
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
            att = np.where(blocked[:, None], -np.inf, att)
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(batch, length, d)
            out = out @ w[f"attns.{i}.out_proj.weight"].T + w[f"attns.{i}.out_proj.bias"]
            y = _layer_norm(q_in + out, w[f"ffn_norms.{i}.weight"], w[f"ffn_norms.{i}.bias"])
            ff = np.maximum(y @ w[f"ffns.{i}.net.0.weight"].T + w[f"ffns.{i}.net.0.bias"], 0.0)
            ff = ff @ w[f"ffns.{i}.net.3.weight"].T + w[f"ffns.{i}.net.3.bias"]
            x = (y + ff) * keep
        return _layer_norm(x, w["last_norm.weight"], w["last_norm.bias"])

    def score(self, users):
        seqs = [self.sequences[u][-self.max_len :] + 1 for u in np.asarray(users)]  # ids shifted by one
        length = max(max((len(s) for s in seqs), default=1), 1)
        padded = np.zeros((len(seqs), length), dtype=np.int64)
        for r, s in enumerate(seqs):
            if len(s):
                padded[r, -len(s) :] = s
        last = self.hidden(padded)[:, -1, :]
        return (last @ self.w["item_emb.weight"][1:].T).astype(np.float32)
