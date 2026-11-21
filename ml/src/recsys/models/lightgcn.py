"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (He et al., SIGIR 2020).

Users and items are nodes of one bipartite graph, with an edge for every interaction. Each node starts
with a learned embedding e^(0). One propagation layer replaces every embedding by the normalized sum of
its neighbours' embeddings:

    e^(k+1) = A_hat e^(k),   A_hat = D^(-1/2) A D^(-1/2),   A = [[0, R], [R^T, 0]]

After K layers the final embedding is the mean of all layers, (e^(0) + ... + e^(K)) / (K + 1), and
score(u, i) = e_u . e_i. After 2 layers a user's embedding contains items liked by *similar users*
(multi-hop collaborative signal). Unlike its predecessor NGCF, LightGCN has no weight matrices and no
non-linearities: only the propagation. It is trained with the BPR loss, with L2 on the layer-0
embeddings.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator
from recsys.models.bpr import positive_keys, sample_negatives


def normalized_adjacency(X: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetrically normalized adjacency D^-1/2 A D^-1/2 of the user-item bipartite graph."""
    n_users, n_items = X.shape
    R = X.astype(np.float32).tocsr()
    A = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D = sp.diags(inv_sqrt)
    assert A.shape == (n_users + n_items, n_users + n_items)
    return (D @ A @ D).tocsr()


def to_torch_sparse(M: sp.csr_matrix) -> torch.Tensor:
    M = M.tocsr()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        return torch.sparse_csr_tensor(
            torch.from_numpy(M.indptr.astype(np.int64)),
            torch.from_numpy(M.indices.astype(np.int64)),
            torch.from_numpy(M.data.astype(np.float32)),
            size=M.shape,
            check_invariants=False,
        )


class SymmetricSpMM(torch.autograd.Function):
    """y = A @ x for a *symmetric* sparse A, with a fast backward pass.

    The gradient w.r.t. x is A^T g = A g (A is symmetric), so the backward pass is one more sparse
    multiply with the same matrix. PyTorch's generic sparse backward is ~11x slower on CPU
    (measured: 647 ms vs 58 ms for 3 layers on ML-1M), which made LightGCN impractical without a GPU.
    """

    @staticmethod
    def forward(ctx, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.adj = adj
        return adj @ x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, ctx.adj @ grad


class LightGCN(IterativeRecommender):
    name = "lightgcn"
    # The paper uses batch 2048 / lr 1e-3. Every step re-propagates the whole graph, so on a CPU we use
    # bigger batches (4x fewer propagations per epoch) with a slightly higher learning rate.
    defaults: ClassVar[dict[str, Any]] = {
        "dim": 64,
        "layers": 3,
        "lr": 2e-3,
        "reg": 1e-4,
        "batch_size": 8192,
        "epochs": 1000,
        "eval_every": 10,
        "patience": 5,
        "seed": 42,
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        torch.manual_seed(p["seed"])
        self._rng = np.random.default_rng(p["seed"])
        X = data.matrix.tocsr()
        self.n_users_, self.n_items_ = X.shape
        self._adj = to_torch_sparse(normalized_adjacency(X))
        self._users = np.repeat(np.arange(self.n_users_, dtype=np.int64), np.diff(X.indptr))
        self._items = X.indices.astype(np.int64)
        self._pos_keys = positive_keys(X)

        self.emb_ = torch.nn.Parameter(torch.randn(self.n_users_ + self.n_items_, p["dim"]) * 0.1)
        self._optimizer = torch.optim.Adam([self.emb_], lr=p["lr"])
        self._final = None
        self._train(validator)
        del self._optimizer, self._users, self._items, self._pos_keys
        return self

    def propagate(self) -> torch.Tensor:
        """Final embeddings: the mean of layers 0..K of graph propagation."""
        layer = self.emb_
        total = layer
        for _ in range(self.params["layers"]):
            layer = SymmetricSpMM.apply(self._adj, layer)
            total = total + layer
        return total / (self.params["layers"] + 1)

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        self._final = None  # cached embeddings are stale once the weights change
        order = self._rng.permutation(self._users.size)
        total = 0.0
        for s in range(0, order.size, p["batch_size"]):
            b = order[s : s + p["batch_size"]]
            u = torch.from_numpy(self._users[b])
            i = torch.from_numpy(self._items[b] + self.n_users_)
            j = torch.from_numpy(
                sample_negatives(self._users[b], self._pos_keys, self.n_items_, self._rng) + self.n_users_
            )
            final = self.propagate()  # the graph must be re-propagated after every update
            margin = (final[u] * (final[i] - final[j])).sum(dim=1)
            e0 = self.emb_
            l2 = (e0[u].pow(2).sum() + e0[i].pow(2).sum() + e0[j].pow(2).sum()) / (2 * b.size)
            loss = -F.logsigmoid(margin).mean() + p["reg"] * l2
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total += loss.item() * b.size
        return total / order.size

    def _snapshot(self):
        return self.emb_.detach().clone()

    def _restore(self, state) -> None:
        with torch.no_grad():
            self.emb_.copy_(state)
        self._final = None

    def __getstate__(self) -> dict:
        """Pickle the propagated final embeddings (all that scoring needs), not the sparse graph."""
        state = dict(self.__dict__)
        if state.get("_final") is None and "_adj" in state:
            with torch.no_grad():
                state["_final"] = self.propagate()
        state.pop("_adj", None)
        return state

    def score(self, users: np.ndarray) -> np.ndarray:
        if self._final is None:
            with torch.no_grad():
                self._final = self.propagate()
        u = self._final[torch.as_tensor(np.asarray(users), dtype=torch.long)]
        return (u @ self._final[self.n_users_ :].T).numpy()
