"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

A user's history is read as an ordered sequence of items, like words in a sentence. A causal
Transformer (each position may only look at itself and earlier positions) turns the sequence into a
hidden state per position; the state after item t is trained to predict item t+1:

    score(next = i | s_1..s_t) = h_t . e_i        (e_i is the item's input embedding, shared)

``loss="bce"`` is the paper's objective (the true next item vs one sampled negative per position).
``loss="ce"`` uses a full softmax over all items instead, which later work found much stronger
(Klenitskiy & Vasilev, RecSys 2023) and is cheap for a catalog of a few thousand items.

Sequences are left-padded with 0 (item ids are shifted by +1), so the last position always holds the
most recent item; to recommend, the state at the last position is scored against every item.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator
from recsys.models.bpr import positive_keys, sample_negatives


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.Dropout(dropout), nn.ReLU(), nn.Linear(dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, dim: int, max_len: int, blocks: int, heads: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.attn_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-8) for _ in range(blocks))
        self.attns = nn.ModuleList(
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) for _ in range(blocks)
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-8) for _ in range(blocks))
        self.ffns = nn.ModuleList(PointWiseFeedForward(dim, dropout) for _ in range(blocks))
        self.last_norm = nn.LayerNorm(dim, eps=1e-8)
        self.dim, self.heads, self.max_len = dim, heads, max_len
        causal = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)
        # Small Xavier-normal weights, as in the original implementation. PyTorch's default N(0, 1)
        # embeddings make the initial logits ~±64 and training fails (measured: val NDCG 0.005).
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()  # padding id

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, L) item ids + 1, left-padded with 0, L <= max_len. Returns (batch, L, dim).

        Padding never receives attention, and positions are counted from the right (the most recent item
        always has position max_len - 1), so a sequence gets the same hidden states whether its batch is
        padded to 200 or trimmed to its longest member. That lets training trim every batch.
        """
        length = seq.shape[1]
        pad = seq == 0
        keep = (~pad).unsqueeze(-1)
        x = self.item_emb(seq) * self.dim**0.5 + self.pos_emb.weight[self.max_len - length :]
        x = self.drop(x) * keep
        eye = torch.eye(length, dtype=torch.bool)
        # True = blocked: future positions, and padding keys (each position may still see itself, so
        # all-padding rows do not produce NaN; their outputs are zeroed by `keep` anyway).
        blocked = self.causal_mask[:length, :length] | (pad[:, None, :] & ~eye)
        blocked = blocked.repeat_interleave(self.heads, dim=0)  # (batch * heads, L, L)
        for norm, attn, ffn_norm, ffn in zip(
            self.attn_norms, self.attns, self.ffn_norms, self.ffns, strict=True
        ):
            q = norm(x)
            out, _ = attn(q, x, x, attn_mask=blocked, need_weights=False)
            x = ffn(ffn_norm(q + out)) * keep
        return self.last_norm(x)


def left_padded(sequences: list[np.ndarray], length: int) -> np.ndarray:
    """Stack sequences (item ids + 1) into a (n, length) array, keeping the most recent items."""
    out = np.zeros((len(sequences), length), dtype=np.int64)
    for r, seq in enumerate(sequences):
        tail = seq[-length:]
        if tail.size:
            out[r, -tail.size :] = tail
    return out


class SASRec(IterativeRecommender):
    name = "sasrec"
    defaults: ClassVar[dict[str, Any]] = {
        "dim": 64,
        "blocks": 2,
        "heads": 1,
        "max_len": 200,
        "dropout": 0.2,
        "loss": "bce",
        "order": "time",  # "shuffled" = ablation: each user's history in random order (a set, not a sequence)
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 300,
        "eval_every": 5,
        "patience": 5,
        "seed": 42,
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        if p["loss"] not in ("bce", "ce"):
            raise ValueError("loss must be 'bce' or 'ce'")
        if p["order"] not in ("time", "shuffled"):
            raise ValueError("order must be 'time' or 'shuffled'")
        torch.manual_seed(p["seed"])
        self._rng = np.random.default_rng(p["seed"])
        X = data.matrix.tocsr()
        n_users, self.n_items_ = X.shape
        # Interactions are stored chronologically within each user (see data/split.py).
        log = data.interactions
        by_user = log.groupby("user_idx", sort=True)["item_idx"].apply(lambda s: s.to_numpy() + 1)
        sequences = [by_user.get(u, np.empty(0, dtype=np.int64)) for u in range(n_users)]
        if p["order"] == "shuffled":
            sequences = [self._rng.permutation(s) for s in sequences]
        L = p["max_len"]
        self._inputs = left_padded([s[:-1] for s in sequences], L)  # predict s[t+1] from s[..t]
        self._targets = left_padded([s[1:] for s in sequences], L)
        self.history_seq_ = left_padded(sequences, L)  # full history, used for scoring
        self._lengths = (self._targets != 0).sum(axis=1)  # training positions per user
        self._trainable = np.flatnonzero(self._lengths > 0)  # users with >= 2 items
        self._pos_keys = positive_keys(X)

        self.net_ = SASRecNet(self.n_items_, p["dim"], L, p["blocks"], p["heads"], p["dropout"])
        self._optimizer = torch.optim.Adam(self.net_.parameters(), lr=p["lr"], betas=(0.9, 0.98))
        self._train(validator)
        del self._optimizer, self._inputs, self._targets, self._pos_keys, self._lengths
        return self

    def _length_batches(self) -> list[np.ndarray]:
        """Batches of users with similar sequence lengths (randomized each epoch), in random order.
        Attention costs O(L^2), so trimming each batch to its longest sequence saves most of the work."""
        users = self._trainable
        noisy = self._lengths[users] * self._rng.uniform(0.9, 1.1, users.size)  # vary groupings per epoch
        by_length = users[np.argsort(noisy, kind="stable")]
        size = self.params["batch_size"]
        batches = [by_length[s : s + size] for s in range(0, by_length.size, size)]
        return [batches[i] for i in self._rng.permutation(len(batches))]

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        self.net_.train()
        total, count = 0.0, 0
        item_table = self.net_.item_emb.weight
        for b in self._length_batches():
            length = int(self._lengths[b].max())
            targets = torch.from_numpy(self._targets[b, -length:])
            mask = targets != 0
            hidden = self.net_(torch.from_numpy(self._inputs[b, -length:]))[mask]  # (positions, dim)
            pos = targets[mask]
            if p["loss"] == "ce":
                logits = hidden @ item_table[1:].T  # all real items (ids shifted by one)
                loss = F.cross_entropy(logits, pos - 1)
            else:
                users = np.repeat(b, mask.sum(dim=1).numpy())
                neg = torch.from_numpy(sample_negatives(users, self._pos_keys, self.n_items_, self._rng) + 1)
                pos_logit = (hidden * item_table[pos]).sum(-1)
                neg_logit = (hidden * item_table[neg]).sum(-1)
                loss = F.binary_cross_entropy_with_logits(
                    pos_logit, torch.ones_like(pos_logit)
                ) + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total += loss.item() * pos.numel()
            count += pos.numel()
        return total / max(count, 1)

    def _snapshot(self):
        return {k: v.detach().clone() for k, v in self.net_.state_dict().items()}

    def _restore(self, state) -> None:
        self.net_.load_state_dict(state)

    def score(self, users: np.ndarray) -> np.ndarray:
        self.net_.eval()
        seq = self.history_seq_[np.asarray(users)]
        length = max(int((seq != 0).sum(axis=1).max()), 1)
        with torch.no_grad():
            hidden = self.net_(torch.from_numpy(seq[:, -length:]))
            last = hidden[:, -1, :]  # state after the most recent item
            return (last @ self.net_.item_emb.weight[1:].T).numpy()
