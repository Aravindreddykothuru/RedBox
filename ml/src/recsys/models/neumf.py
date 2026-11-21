"""NeuMF: Neural Collaborative Filtering (He et al., WWW 2017).

Two branches see the same (user, item) pair:
  * GMF (generalized matrix factorization): element-wise product p_u * q_i, a learnable dot product
  * MLP: concatenation [p'_u ; q'_i] through ReLU layers, which can learn non-linear interactions
Their outputs are concatenated and mapped to one logit. Training is binary classification:
liked pairs are 1, and ``negatives`` randomly sampled unobserved items per positive are 0.

Because the MLP must run on every (user, item) pair, scoring the full catalog is far more expensive
than a dot product, so scoring runs in chunks of users.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, ClassVar, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator
from recsys.models.bpr import positive_keys, sample_negatives


class NeuMFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, gmf_dim: int, mlp_dims: list[int]):
        super().__init__()
        if mlp_dims[0] % 2:
            raise ValueError("mlp_dims[0] must be even: it is the concatenation of two embeddings")
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dims[0] // 2)
        self.mlp_item = nn.Embedding(n_items, mlp_dims[0] // 2)
        layers: list[nn.Module] = []
        for a, b in pairwise(mlp_dims):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + mlp_dims[-1], 1)
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(u) * self.gmf_item(i)
        mlp = self.mlp(torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class NeuMF(IterativeRecommender):
    name = "neumf"
    defaults: ClassVar[dict[str, Any]] = {
        "gmf_dim": 32,
        "mlp_dims": [64, 32, 16],
        "negatives": 4,
        "lr": 1e-3,
        "batch_size": 2048,
        "epochs": 50,
        "eval_every": 2,
        "patience": 3,
        "seed": 42,
        "score_chunk": 64,  # users scored at once (memory: chunk x n_items pairs through the MLP)
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        torch.manual_seed(p["seed"])
        self._rng = np.random.default_rng(p["seed"])
        X = data.matrix.tocsr()
        n_users, self.n_items_ = X.shape
        self._users = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(X.indptr))
        self._items = X.indices.astype(np.int64)
        self._pos_keys = positive_keys(X)
        self.net_ = NeuMFNet(n_users, self.n_items_, p["gmf_dim"], list(p["mlp_dims"]))
        self._optimizer = torch.optim.Adam(self.net_.parameters(), lr=p["lr"])
        self._train(validator)
        del self._optimizer, self._users, self._items, self._pos_keys
        return self

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        self.net_.train()
        k = p["negatives"]
        # Each positive pair plus k sampled negatives for the same user, shuffled together.
        users = np.concatenate([self._users] + [self._users] * k)
        negs = [sample_negatives(self._users, self._pos_keys, self.n_items_, self._rng) for _ in range(k)]
        items = np.concatenate([self._items, *negs])
        labels = np.concatenate([np.ones(self._users.size), np.zeros(self._users.size * k)]).astype(
            np.float32
        )
        order = self._rng.permutation(users.size)
        total = 0.0
        for s in range(0, order.size, p["batch_size"]):
            b = order[s : s + p["batch_size"]]
            logits = self.net_(torch.from_numpy(users[b]), torch.from_numpy(items[b]))
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels[b]))
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total += loss.item() * b.size
        return total / order.size

    def _snapshot(self):
        return {k: v.detach().clone() for k, v in self.net_.state_dict().items()}

    def _restore(self, state) -> None:
        self.net_.load_state_dict(state)

    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        self.net_.eval()
        out = np.empty((users.size, self.n_items_), dtype=np.float32)
        all_items = torch.arange(self.n_items_)
        chunk = self.params["score_chunk"]
        with torch.no_grad():
            for s in range(0, users.size, chunk):
                block = torch.as_tensor(users[s : s + chunk], dtype=torch.long)
                u = block.repeat_interleave(self.n_items_)
                i = all_items.repeat(block.numel())
                out[s : s + block.numel()] = self.net_(u, i).view(block.numel(), self.n_items_).numpy()
        return out
