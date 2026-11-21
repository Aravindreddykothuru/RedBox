"""Mult-VAE: Variational Autoencoders for Collaborative Filtering (Liang et al., WWW 2018).

The input is a user's whole interaction vector x_u (one entry per item). An encoder compresses it into
a Gaussian latent distribution N(mu, sigma^2) over a 200-dimensional "taste" vector z, and a decoder
expands z back into a probability distribution over all items (softmax = multinomial likelihood):

    loss = -sum_i x_ui * log softmax(f(z))_i  +  beta * KL( q(z | x_u) || N(0, I) )

The KL weight beta is annealed from 0 up to ``anneal_cap`` (the paper found partial regularization,
beta < 1, works best). At prediction time z = mu (no sampling) and the decoder logits are the scores.
Scoring only needs the user's interaction vector, so new users can be scored without retraining.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.split import TrainData
from recsys.models.base import IterativeRecommender, Validator


class MultVAENet(nn.Module):
    def __init__(self, n_items: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.enc = nn.Linear(n_items, hidden_dim)
        self.enc_out = nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log-variance
        self.dec = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, n_items)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.drop(F.normalize(x, dim=1))  # L2-normalized input, dropout as input corruption
        h = torch.tanh(self.enc(h))
        mu, logvar = self.enc_out(h).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        logits = self.dec_out(torch.tanh(self.dec(z)))
        return logits, mu, logvar


class MultVAE(IterativeRecommender):
    name = "multvae"
    defaults: ClassVar[dict[str, Any]] = {
        "hidden_dim": 600,
        "latent_dim": 200,
        "dropout": 0.5,
        "anneal_cap": 0.2,
        "anneal_epochs": 100,  # epochs over which beta rises linearly from 0 to anneal_cap
        "lr": 1e-3,
        "batch_size": 500,
        "epochs": 400,
        "eval_every": 5,
        "patience": 5,
        "seed": 42,
    }

    def fit(self, data: TrainData, validator: Validator | None = None) -> Self:
        p = self.params
        torch.manual_seed(p["seed"])
        self._rng = np.random.default_rng(p["seed"])
        self.X_ = data.matrix.tocsr().astype(np.float32)
        self.net_ = MultVAENet(self.X_.shape[1], p["hidden_dim"], p["latent_dim"], p["dropout"])
        self._optimizer = torch.optim.Adam(self.net_.parameters(), lr=p["lr"])
        self._steps_per_epoch = int(np.ceil(self.X_.shape[0] / p["batch_size"]))
        self._step = 0
        self._train(validator)
        del self._optimizer
        return self

    def _beta(self) -> float:
        total = self.params["anneal_epochs"] * self._steps_per_epoch
        return self.params["anneal_cap"] * min(1.0, self._step / max(total, 1))

    def _run_epoch(self, epoch: int) -> float:
        p = self.params
        self.net_.train()
        order = self._rng.permutation(self.X_.shape[0])
        total = 0.0
        for s in range(0, order.size, p["batch_size"]):
            b = order[s : s + p["batch_size"]]
            x = torch.from_numpy(self.X_[b].toarray())
            logits, mu, logvar = self.net_(x)
            nll = -(F.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = nll + self._beta() * kl
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._step += 1
            total += loss.item() * b.size
        return total / order.size

    def _snapshot(self):
        return {k: v.detach().clone() for k, v in self.net_.state_dict().items()}

    def _restore(self, state) -> None:
        self.net_.load_state_dict(state)

    def score(self, users: np.ndarray) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.X_[np.asarray(users)].toarray())
            return self.net_(x)[0].numpy()
