"""The recommendation engine behind the API: everything is computed from the user's current history,
so a rating changes the very next response (no retraining, no per-user state)."""

from __future__ import annotations

import functools
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.hybrid.features import Context, FeatureTable, build_features, item_cosine
from recsys.hybrid.ranker import mmr_order
from recsys.serving.bundle import load_ranker, load_scorers

_WORD = re.compile(r"[a-z0-9]+")

# Plain names for the generators and the ranker's signals ("Why this pick?", Engine.why).
GENERATOR_LABELS = {
    "sasrec_ce": "SASRec",
    "sasrec": "SASRec",
    "ease": "EASE",
    "ials": "iALS",
    "markov": "Markov chain",
    "mostpop": "Popularity",
    "lightgcn": "LightGCN",
}
_SIGNALS = {
    "sasrec_ce": "SASRec: the order you watched in",
    "sasrec": "SASRec: the order you watched in",
    "ease": "EASE: films liked together",
    "ials": "iALS: hidden taste factors",
    "markov": "Markov chain: what usually comes next",
    "lightgcn": "LightGCN: the viewer-film graph",
    "mostpop": "How popular it is",
    "item_log_pop": "How popular it is",
    "item_pop_vs_user": "How popular it is",
    "user_mean_log_pop": "How popular it is",
    "n_generators": "How many models agree",
    "cos_to_last": "Like the last film you liked",
    "genre_affinity": "Your genre taste",
    "genre_": "Your genre taste",
    "item_year": "When it came out",
    "user_log_history": "How much you have rated",
}


def one_at_a_time(method):
    """Run an engine method under the engine's lock. The scorers keep the history being scored as state
    (set_history), so two requests scoring at once in the API's worker threads could each be scored with
    the other's history; the lock makes them take turns (each takes milliseconds)."""

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


def signal_of(feature: str) -> str:
    """The plain-language source of one ranker feature (e.g. "ease_z" -> "EASE: films liked together")."""
    for prefix, label in sorted(_SIGNALS.items(), key=lambda kv: -len(kv[0])):
        if feature == prefix or feature.startswith(prefix):
            return label
    return feature


@dataclass
class Recommendation:
    item_idx: int
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    score: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_matrix(users: dict, n_items: int) -> sp.csr_matrix:
    """Binary (users x items) matrix rebuilt from the bundle's dataset_users.npz."""
    n_users = len(users["user_ids"])
    rows = np.repeat(np.arange(n_users), np.diff(users["indptr"]))
    data = np.ones(rows.size, dtype=np.float32)
    return sp.csr_matrix((data, (rows, users["items"])), shape=(n_users, n_items))


def item_statistics(users: dict, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    M = dataset_matrix(users, n_items)
    return item_cosine(M), np.asarray(M.sum(axis=0)).ravel()


def write_item_stats(bundle: Path) -> Path:
    """Save the ranker's global item statistics (item-item cosine, popularity) next to the bundle's
    dataset_users.npz, so a public deployment can drop the raw MovieLens histories (which may not be
    redistributed) and still compute exactly the same features."""
    bundle = Path(bundle)
    users = dict(np.load(bundle / "dataset_users.npz"))
    n_items = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["n_items"]
    sim, pop = item_statistics(users, n_items)
    out = bundle / "item_stats.npz"
    np.savez_compressed(out, sim=sim, pop=pop)
    return out


class Engine:
    def __init__(self, bundle: Path, encoder=None):
        self._lock = threading.RLock()  # see one_at_a_time
        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        self.items = pd.read_parquet(self.bundle / "items.parquet")
        self.items["genres"] = self.items["genres"].map(list)
        self.n_items = len(self.items)
        self.scorers = load_scorers(self.bundle, self.manifest)
        self.ranker = load_ranker(self.bundle)
        nb = np.load(self.bundle / "neighbours.npz")
        self.nb_idx, self.nb_sim = nb["idx"], nb["sim"]
        self.seeds = np.load(self.bundle / "seeds.npy")
        genres_path = self.bundle / "genres.npz"
        self.genres = sp.load_npz(genres_path).tocsr() if genres_path.exists() else None
        self.genres_dense = self.genres.toarray() if self.genres is not None else None
        self.genre_names: list[str] = self.manifest.get("genre_names", [])
        years = self.items["year"]
        self.years = years.astype("float64").to_numpy() if years.notna().any() else None
        self.popularity = self.items["popularity"].to_numpy(np.float32)
        self.movie_to_idx = dict(
            zip(self.items["movie_id"].tolist(), self.items["item_idx"].tolist(), strict=True)
        )
        self.n_candidates = int(self.manifest["candidates_per_generator"])
        users_path = self.bundle / "dataset_users.npz"
        self._dataset_users = dict(np.load(users_path)) if users_path.exists() else None
        # The crowd forecast (`recsys forecast`, serving/forecast.py): what the hybrid predicts viewers will
        # watch next, per film and per genre of taste. Drives trending and brand-new users; bundles without
        # it fall back to popularity.
        forecast_path = self.bundle / "forecast.npz"
        forecast = dict(np.load(forecast_path)) if forecast_path.exists() else {}
        self.crowd: np.ndarray | None = forecast.get("crowd")
        self.crowd_by_genre: np.ndarray | None = forecast.get("by_genre")
        offline = self.bundle / "offline_results.csv"
        self.offline_results = pd.read_csv(offline) if offline.exists() else None
        # Global item statistics for the ranker's features, from the same interactions the export used
        # (train + validation). Computing them from one user's history would give different features than
        # the ranker was trained and evaluated with (train/serve skew).
        # Without dataset_users.npz (public deployments: no raw MovieLens histories), the precomputed
        # item_stats.npz gives the same numbers, and the Model Lab only offers the signed-in user.
        stats_path = self.bundle / "item_stats.npz"
        if self._dataset_users is not None:
            self.item_sim, self.item_pop = item_statistics(self._dataset_users, self.n_items)
        elif stats_path.exists():
            stats = np.load(stats_path)
            self.item_sim, self.item_pop = stats["sim"], stats["pop"]
        else:
            raise ValueError(f"{self.bundle} needs dataset_users.npz or item_stats.npz for item statistics")
        # Optional content side (`recsys content`): one sentence embedding per movie + the query encoder.
        text_path = self.bundle / "text_embeddings.npy"
        self.text_emb = np.load(text_path) if text_path.exists() else None
        # The query encoder (anything with .encode(list[str]) -> unit vectors). By default the bundle's
        # ONNX model, loaded on the first search so startup stays fast; tests pass a stand-in.
        self.encoder = encoder
        if "keywords" in self.items:
            self.items["keywords"] = self.items["keywords"].map(list)
        # New films outside the dataset (`recsys newfilms`, serving/newfilms.py, and the API's daily refresh):
        # appended after the model's films, so they can be shown, searched, rated and found by "similar
        # story", but no generator scores them; new_films() ranks them by their iALS factors. Those start as
        # the guess the film's story gives (new_prior) and move towards what its raters like as ratings
        # arrive (set_feedback). self.n_items stays the number of films the models know.
        self.new_prior: np.ndarray | None = None
        self.new_factors: np.ndarray | None = None
        self.new_likes = np.zeros(0, dtype=np.int64)
        self.new_strength = float(self.manifest.get("new_items", {}).get("prior_strength", 10.0))
        new_path = self.bundle / "new_items.parquet"
        if new_path.exists():
            text = np.load(self.bundle / "new_items_text.npy") if self.text_emb is not None else None
            self.add_new_films(
                pd.read_parquet(new_path), np.load(self.bundle / "new_items_factors.npy"), text
            )

    @property
    def has_text(self) -> bool:
        return self.text_emb is not None and (
            self.encoder is not None or (self.bundle / "text_model" / "model.onnx").exists()
        )

    def encode_query(self, text: str) -> np.ndarray:
        if not self.has_text:
            raise ValueError("this bundle has no text model: run `recsys content <dataset> --bundle ...`")
        if self.encoder is None:
            from recsys.content.encoder import TextEncoder

            self.encoder = TextEncoder(self.bundle / "text_model", max_length=64)
        return self.encoder.encode([text])[0]

    def dataset_user_ids(self) -> list[int]:
        return [] if self._dataset_users is None else self._dataset_users["user_ids"].tolist()

    def dataset_history(self, user_id: int) -> list[int]:
        """Chronological liked items of a user from the training dataset (Model Lab)."""
        if self._dataset_users is None:
            raise KeyError("this bundle has no dataset users")
        pos = np.flatnonzero(self._dataset_users["user_ids"] == user_id)
        if not pos.size:
            raise KeyError(f"unknown dataset user {user_id}")
        start, end = self._dataset_users["indptr"][pos[0]], self._dataset_users["indptr"][pos[0] + 1]
        return self._dataset_users["items"][start:end].tolist()

    # ---------------------------------------------------------------------------------- helpers
    @property
    def models(self) -> list[str]:
        return ["hybrid", *self.scorers]

    def to_idx(self, movie_ids) -> list[int]:
        return [self.movie_to_idx[m] for m in movie_ids if m in self.movie_to_idx]

    def _rec(self, i: int, score: float, reason: str) -> Recommendation:
        row = self.items.iloc[int(i)]
        year = None if pd.isna(row["year"]) else int(row["year"])
        return Recommendation(
            int(i), int(row["movie_id"]), str(row["title"]), year, list(row["genres"]), float(score), reason
        )

    def _history(self, history: list[int]) -> sp.csr_matrix:
        data = np.ones(len(set(history)), dtype=np.float32)
        return sp.csr_matrix((data, ([0] * len(data), sorted(set(history)))), shape=(1, self.n_items))

    def _attach(self, history: list[int]) -> sp.csr_matrix:
        H = self._history(history)
        seq = [np.asarray(history, dtype=np.int64)]
        for s in self.scorers.values():
            s.set_history(H, seq)
        return H

    def _top(self, scores: np.ndarray, exclude: set[int], k: int) -> np.ndarray:
        scores = scores.astype(np.float64).copy()
        exclude = [i for i in exclude if i < len(scores)] if exclude else []
        if exclude:
            scores[exclude] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        return top[np.argsort(-scores[top], kind="stable")]

    # ---------------------------------------------------------------------------------- explanations
    @one_at_a_time
    def explain(self, item: int, history: list[int]) -> str:
        if not history:
            return "Popular with viewers right now"
        liked = set(history)
        nbrs, sims = self.nb_idx[item], self.nb_sim[item]
        hits = [(s, n) for n, s in zip(nbrs, sims, strict=True) if n in liked and s >= 0.1]
        if hits:
            _, best = max(hits)
            return f"Because you liked {self.items.iloc[int(best)]['title']}"
        if self.genres is not None:
            profile = np.asarray(self._history(history) @ self.genres_dense).ravel()
            shared = [(profile[j], self.genre_names[j]) for j in self.genres[item].indices if profile[j] > 0]
            if shared:
                return f"Matches your taste for {max(shared)[1]}"
        return "Popular with viewers who like what you like"

    @one_at_a_time
    def why(self, item: int, history: list[int]) -> dict | None:
        """How the ranker scored one film for this history: the exact contribution of every feature
        (TreeSHAP, LightGBM's pred_contrib: they add up to the score), summed by where each signal comes
        from, plus the film's rank among the candidates and which generators proposed it. None when the
        film is not among the user's candidates, so the ranker never scored it."""
        if not history or item >= self.n_items:
            return None
        table = self.features(history)
        rows = np.flatnonzero(table.items == item)
        if not len(rows):
            return None
        row = int(rows[0])
        contrib = self.ranker.predict(table.X[row : row + 1], pred_contrib=True)[0]
        groups: dict[str, float] = {}
        for name, value in zip(table.names, contrib[:-1], strict=True):
            label = signal_of(name)
            groups[label] = groups.get(label, 0.0) + float(value)
        total = sum(abs(v) for v in groups.values()) or 1.0
        parts = sorted(
            (
                {"signal": g, "contribution": v, "share": v / total}
                for g, v in groups.items()
                if abs(v) > 1e-9
            ),
            key=lambda p: -abs(p["contribution"]),
        )
        scores = self.ranker.predict(table.X)
        proposed = [
            GENERATOR_LABELS.get(n.removesuffix("_in_top"), n.removesuffix("_in_top"))
            for n, v in zip(table.names, table.X[row], strict=True)
            if n.endswith("_in_top") and v > 0
        ]
        return {
            "score": float(contrib.sum()),
            "baseline": float(contrib[-1]),
            "rank": int((scores > scores[row]).sum()) + 1,
            "candidates": len(table.items),
            "proposed_by": proposed,
            "parts": parts,
        }

    @one_at_a_time
    def features(self, history: list[int], extra_items: np.ndarray | None = None) -> FeatureTable:
        """The ranker's candidate/feature table for one history (row user index 0); ``extra_items`` are
        scored too even if no generator proposed them."""
        H = self._attach(history)
        ctx = Context(
            H,
            np.array([history[-1]]),
            self.genres,
            self.years,
            None,
            item_pop=self.item_pop,
            item_sim=self.item_sim,
        )
        table = build_features(
            self.scorers, np.array([0]), ctx, n_candidates=self.n_candidates, extra_items=extra_items
        )
        if table.names != self.manifest["feature_names"]:
            raise RuntimeError("feature layout differs from the one the ranker was trained on")
        return table

    GROUP_STRATEGIES = ("average", "least_misery", "most_pleasure")

    @one_at_a_time
    def group(
        self,
        histories: list[list[int]],
        strategy: str = "average",
        k: int = 10,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float, list[float]]]:
        """Films for several people watching together: [(item, group score, each member's fit 0-1)].

        Every member's hybrid candidates are pooled, and the ranker then scores the whole pool for every
        member, so all fits are about the same films. Each member's scores are rescaled to 0-1 within the
        pool, so everyone counts the same, and combined by ``strategy``:
        - "average": the most satisfaction overall;
        - "least_misery": the least happy member decides, so nobody hates the pick;
        - "most_pleasure": the happiest member decides.
        A member without history counts with the crowd forecast (what many viewers watch next).
        Nobody's watched or rated films (``exclude``) are proposed."""
        if strategy not in self.GROUP_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.GROUP_STRATEGIES}")
        blocked = set(exclude or ()) | {i for h in histories for i in h}
        pool: set[int] = set()
        for h in histories:
            if h:
                pool |= {int(i) for i in self.features(h).items}
        if not pool:  # nobody has a history: the crowd forecast's top films
            crowd = self.crowd if self.crowd is not None else self.popularity
            pool = {int(i) for i in self._top(crowd, blocked, 150)}
        items = np.array(sorted(pool - blocked), dtype=np.int64)
        if not len(items):
            return []
        fits = []
        for h in histories:
            if h:
                table = self.features(h, extra_items=items)
                score = dict(zip(table.items.tolist(), self.ranker.predict(table.X).tolist(), strict=True))
                s = np.array([score.get(int(i), np.nan) for i in items], dtype=np.float64)
            else:
                base = self.crowd if self.crowd is not None else self.popularity
                s = base[items].astype(np.float64)
            s = np.where(np.isnan(s), np.nanmin(s), s)
            span = s.max() - s.min()
            fits.append((s - s.min()) / span if span > 0 else np.full_like(s, 0.5))
        U = np.vstack(fits)
        combined = {"average": U.mean(0), "least_misery": U.min(0), "most_pleasure": U.max(0)}[strategy]
        order = np.argsort(-combined, kind="stable")[:k]
        return [(int(items[j]), float(combined[j]), [float(x) for x in U[:, j]]) for j in order]

    # ---------------------------------------------------------------------------------- endpoints
    @one_at_a_time
    def recommend(
        self,
        history: list[int],
        exclude: set[int] | None = None,
        k: int = 20,
        lam: float = 1.0,
        model: str = "hybrid",
        genres: list[str] | None = None,
    ) -> list[Recommendation]:
        """Top-k for a user whose liked items (chronological) are ``history``; ``exclude`` = everything the
        user has already rated (liked or not), so nothing they know is ever recommended again. A user with
        no liked film yet gets the cold-start forecast for the ``genres`` they chose."""
        exclude = set(exclude or ()) | set(history)
        if model not in self.models:
            raise ValueError(f"unknown model '{model}'; available: {self.models}")
        if not history:
            return self.cold_start(genres, k=k, exclude=exclude)
        if model != "hybrid":
            self._attach(history)
            scores = self.scorers[model].score(np.array([0]))[0]
            return [self._rec(i, scores[i], self.explain(i, history)) for i in self._top(scores, exclude, k)]

        table = self.features(history)
        pred = self.ranker.predict(table.X)
        keep = np.array([i not in exclude for i in table.items])
        items, pred = table.items[keep], pred[keep]
        if lam < 1.0 and self.genres_dense is not None:
            chosen = mmr_order(items, pred, self.genres_dense, lam, k)
        else:
            chosen = items[np.argsort(-pred, kind="stable")][:k]
        score = dict(zip(items.tolist(), pred.tolist(), strict=True))
        return [self._rec(i, score[int(i)], self.explain(int(i), history)) for i in chosen]

    def similar(self, item: int, k: int = 12, mode: str = "cf") -> list[Recommendation]:
        """Similar movies: "cf" = liked by the same people; "content" = similar story and themes (text
        embeddings of keywords and plot), or shared genres and era when the bundle has no text model."""
        if mode == "cf" and item < self.n_items:
            return [
                self._rec(n, s, "Liked by the same viewers")
                for n, s in zip(self.nb_idx[item][:k], self.nb_sim[item][:k], strict=True)
            ]
        if self.text_emb is not None and np.any(self.text_emb[item]):
            sims = (self.text_emb @ self.text_emb[item]).astype(np.float64)
            sims[item] = -np.inf
            return [self._rec(i, sims[i], "Similar story and themes") for i in self._top(sims, set(), k)]
        if self.genres_dense is None:
            raise ValueError("content similarity needs item genres")
        g = self.genres_dense
        inter = g @ g[item]
        union = g.sum(axis=1) + g[item].sum() - inter
        jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        if self.years is not None:
            gap = np.abs(np.nan_to_num(self.years - self.years[item], nan=50.0))
            jaccard = jaccard - 0.01 * np.minimum(gap, 30)  # prefer the same era
        jaccard[item] = -np.inf
        top = np.argsort(-jaccard)[:k]
        return [self._rec(i, jaccard[i], "Similar genres and era") for i in top]

    def semantic_search(
        self, query: str, k: int = 24, exclude: set[int] | None = None, pop_weight: float = 0.05
    ) -> list[Recommendation]:
        """Natural-language search ("mind-bending space movies"): cosine similarity between the query's
        embedding and every movie's text embedding. A small popularity prior breaks near-ties in favour
        of films people know; the cosine itself is returned as the score."""
        q = self.encode_query(query)
        cosine = self.text_emb @ q
        prior = np.log1p(self.popularity) / max(np.log1p(self.popularity.max()), 1e-9)
        prior = np.concatenate([prior, np.zeros(len(cosine) - len(prior))])  # new films: no popularity yet
        ranked = cosine + pop_weight * prior
        ranked[~np.any(self.text_emb, axis=1)] = -np.inf  # movies without a text
        words = set(_WORD.findall(query.lower()))
        out = []
        for i in self._top(ranked, set(exclude or ()), k):
            keywords = list(self.items.iloc[int(i)]["keywords"]) if "keywords" in self.items else []
            # show the viewer keywords that share a word with the query first
            hits = [kw for kw in keywords if words & set(_WORD.findall(kw))]
            shown = (hits + [kw for kw in keywords if kw not in hits])[:3]
            reason = f"Viewers tag it: {', '.join(shown)}" if shown else "Matches your description"
            out.append(self._rec(i, float(cosine[i]), reason))
        return out

    def _tie_break(self, scores: np.ndarray) -> np.ndarray:
        """Popularity only decides between films the forecast scores the same (e.g. both zero)."""
        return scores.astype(np.float64) + 1e-6 * self.popularity / max(float(self.popularity.max()), 1.0)

    @property
    def new_languages(self) -> list[str]:
        """The new films' languages, in the order they were added (English, Hindi, Telugu, Tamil, ...)."""
        if self.new_factors is None or "language" not in self.items:
            return []
        return [str(x) for x in dict.fromkeys(self.items["language"].iloc[self.n_items :].dropna())]

    def _ials_vector(self, history: list[int]) -> np.ndarray | None:
        """The user's exact iALS vector for this history (the fold-in IALSScorer computes), without
        touching any scorer's state; None without an iALS generator."""
        scorer = self.scorers.get("ials")
        if scorer is None:
            return None
        idx = np.unique([int(i) for i in history if 0 <= int(i) < self.n_items])
        if not idx.size:
            return np.zeros(scorer.Q.shape[1])
        Qu = scorer.Q[idx]
        return np.linalg.solve(
            scorer.gram + scorer.alpha * (Qu.T @ Qu), (1.0 + scorer.alpha) * Qu.sum(axis=0)
        )

    def add_new_films(self, rows: pd.DataFrame, factors: np.ndarray, text: np.ndarray | None) -> None:
        """Append new films (bundle rows, their story-based iALS factors and text vectors) after the ones
        already here: at startup, and when the API's daily refresh finds new releases. Earlier films keep
        their item indices."""
        with self._lock:
            rows = rows.drop(columns=["item_idx"], errors="ignore").reset_index(drop=True)
            rows["genres"] = rows["genres"].map(list)
            rows["keywords"] = (
                rows["keywords"].map(list) if "keywords" in rows else [[] for _ in range(len(rows))]
            )
            if "links" not in rows:
                rows["links"] = 0
            start = len(self.items)
            rows.insert(0, "item_idx", np.arange(start, start + len(rows)))
            factors = np.asarray(factors, dtype=np.float64).reshape(len(rows), -1)
            prior = factors if self.new_prior is None else np.vstack([self.new_prior, factors])
            current = factors.copy() if self.new_factors is None else np.vstack([self.new_factors, factors])
            likes = np.concatenate([self.new_likes, np.zeros(len(rows), dtype=np.int64)])
            items = pd.concat([self.items, rows], ignore_index=True)
            # publish the arrays before the rows that index them: a reader holding the old rows never
            # looks past the end of the new arrays
            self.new_prior, self.new_factors, self.new_likes = prior, current, likes
            if self.text_emb is not None and text is not None:
                self.text_emb = np.vstack([self.text_emb, np.asarray(text, dtype=self.text_emb.dtype)])
            self.items = items
            self.movie_to_idx.update(zip(rows["movie_id"].tolist(), rows["item_idx"].tolist(), strict=True))

    def set_feedback(self, feedback: dict[int, list[tuple[list[int], bool]]]) -> None:
        """What the app's users think of the new films: item index -> one (that user's liked history,
        liked it?) per user who rated it. Each rated film's factors are re-estimated from its story's guess
        and its raters' tastes (newfilms.warm_factors: collaborative topic regression), so a film moves
        towards the viewers who liked it as ratings arrive. Replaces any earlier feedback."""
        from recsys.serving.newfilms import warm_factors

        scorer = self.scorers.get("ials")
        if self.new_prior is None or scorer is None:
            return
        with self._lock:
            factors = self.new_prior.copy()
            likes = np.zeros(len(factors), dtype=np.int64)
            for item, ratings in feedback.items():
                j = int(item) - self.n_items
                if not 0 <= j < len(factors):
                    continue
                likes[j] = sum(1 for _, liked in ratings if liked)
                users, liked = [], []
                for history, is_liked in ratings:
                    p = self._ials_vector(history)
                    if p is not None and p.any():  # a viewer with no known films shows no taste yet
                        users.append(p)
                        liked.append(1.0 if is_liked else 0.0)
                if users:
                    factors[j] = warm_factors(
                        self.new_prior[j],
                        np.array(users),
                        np.array(liked),
                        1.0 + scorer.alpha,
                        self.new_strength,
                    )
            self.new_factors, self.new_likes = factors, likes

    @staticmethod
    def _new_reason(lang, likes: int, fallback: str) -> str:
        where = f"New in {lang}" if isinstance(lang, str) and lang else "New"
        if likes:
            return f"{where}: liked by {likes} Red Box viewer{'s' if likes != 1 else ''}"
        return f"{where}: {fallback}"

    def new_films(
        self,
        history: list[int],
        k: int = 20,
        exclude: set[int] | None = None,
        language: str | None = None,
        genres: list[str] | None = None,
    ) -> list[Recommendation]:
        """New films (released after the models' data, or too new for any ratings), ranked for this user.
        With a history: each film's iALS factors (its story's guess, warmed by the app's ratings) against
        the user's exact iALS vector. A brand-new user, with no history: the most widely known releases,
        those in the genres they picked first. ``language`` keeps one language's films only."""
        items, factors, likes = self.items, self.new_factors, self.new_likes
        if factors is None:
            return []
        n_new = min(len(factors), len(items) - self.n_items, len(likes))
        new = items.iloc[self.n_items : self.n_items + n_new]
        langs = new["language"].to_numpy() if "language" in new else np.full(n_new, None)
        pool = np.arange(n_new) if language is None else np.flatnonzero(langs == language)
        if not len(pool):
            return []
        if history:
            u = self._ials_vector(history)
            if u is None:
                return []
            scores = factors[pool] @ u
            fallback = "its story matches your taste"
        else:
            links = pd.to_numeric(new["links"], errors="coerce").fillna(0).to_numpy() if "links" in new else 0
            known = np.log1p(
                np.asarray(links, dtype=np.float64)[pool] if np.ndim(links) else np.zeros(len(pool))
            )
            known /= max(float(known.max()), 1.0)
            wanted = set(genres or ())
            film_genres = new["genres"].to_numpy()[pool]
            match = np.array([len(wanted & set(g)) / max(len(g), 1) for g in film_genres]) if wanted else 0.0
            years = pd.to_numeric(new["year"], errors="coerce").fillna(0).to_numpy()[pool]
            recent = (years - years.min()) / max(float(years.max() - years.min()), 1.0)
            liked = np.minimum(likes[pool], 10) / 10
            scores = known + 0.8 * match + 0.2 * recent + 0.5 * liked
            matched = np.asarray(match) > 0 if wanted else np.zeros(len(pool), dtype=bool)
        skip = set(exclude or ())
        out = []
        for r in np.argsort(-scores, kind="stable"):
            i = self.n_items + int(pool[r])
            if i in skip:
                continue
            if not history:
                fallback = "matches the genres you picked" if matched[r] else "widely talked about"
            reason = self._new_reason(langs[pool[r]], int(likes[pool[r]]), fallback)
            out.append(self._rec(i, float(scores[r]), reason))
            if len(out) >= k:
                break
        return out

    def trending(self, k: int = 20, exclude: set[int] | None = None) -> list[Recommendation]:
        """The films the hybrid predicts most viewers will watch next (the crowd forecast), not the films
        most viewers already watched. Popularity only for a bundle without a forecast."""
        if self.crowd is None:
            return [
                self._rec(i, self.popularity[i], "Popular with viewers right now")
                for i in self._top(self.popularity, set(exclude or ()), k)
            ]
        scores = self._tie_break(self.crowd)
        return [
            self._rec(i, self.crowd[i], "Predicted next for many viewers")
            for i in self._top(scores, set(exclude or ()), k)
        ]

    def cold_start(
        self, genres: list[str] | None = None, k: int = 20, exclude: set[int] | None = None
    ) -> list[Recommendation]:
        """A brand-new user, before any rating: what the hybrid predicts next for dataset viewers whose
        genre taste matches the genres the user chose ("viewers like you"); for every viewer when they
        chose none. Popularity only for a bundle without a forecast."""
        chosen = [g for g in (genres or []) if g in self.genre_names]
        if self.crowd_by_genre is not None and len(self.crowd_by_genre) and chosen:
            rows = [self.genre_names.index(g) for g in chosen]
            scores = self.crowd_by_genre[rows].sum(axis=0)
            named = " and ".join(chosen) if len(chosen) <= 2 else f"{chosen[0]}, {chosen[1]} and more"
            return [
                self._rec(i, scores[i], f"Predicted for viewers who like {named}")
                for i in self._top(self._tie_break(scores), set(exclude or ()), k)
            ]
        return self.trending(k=k, exclude=exclude)

    @one_at_a_time
    def because_you_liked(
        self, history: list[int], k: int = 12, exclude: set[int] | None = None
    ) -> tuple[Recommendation | None, list[Recommendation]]:
        """A row of movies similar to the user's most recent liked movie."""
        if not history:
            return None, []
        anchor = history[-1]
        seen = set(exclude or ()) | set(history)
        picks = [
            (n, s) for n, s in zip(self.nb_idx[anchor], self.nb_sim[anchor], strict=True) if n not in seen
        ][:k]
        title = self.items.iloc[anchor]["title"]
        return self._rec(anchor, 1.0, ""), [self._rec(n, s, f"Because you liked {title}") for n, s in picks]

    @one_at_a_time
    def hidden_gems(
        self, history: list[int], k: int = 12, exclude: set[int] | None = None
    ) -> list[Recommendation]:
        """Well-rated films outside the most-watched quarter that match the user's taste.

        The hybrid's candidates are mostly well-known films, so gems are ranked over the whole catalogue.
        Eligible: popularity between the 25th and 75th percentile and (when the bundle has rating
        statistics) an average rating of at least 3.7. Score: EASE (long-term taste) plus, when available,
        the cosine between the film's text embedding and the user's mean liked-film embedding, both
        z-scored. CF scores alone are noisy for rarely watched films; the content profile steadies them."""
        if not history:
            return []
        name = "ease" if "ease" in self.scorers else next(iter(self.scorers))
        self._attach(history)

        def z(x: np.ndarray) -> np.ndarray:
            return (x - x.mean()) / (x.std() + 1e-9)

        scores = z(self.scorers[name].score(np.array([0]))[0].astype(np.float64))
        if self.text_emb is not None:
            profile = self.text_emb[history].mean(axis=0)
            if np.any(profile):
                cosine = self.text_emb[: self.n_items] @ (profile / np.linalg.norm(profile))
                scores = scores + z(cosine.astype(np.float64))
        lo, hi = np.percentile(self.popularity, [25, 75])
        eligible = (self.popularity >= lo) & (self.popularity < hi)
        if "avg_rating" in self.items:
            # the models' films only: new films (after n_items) have no ratings and no model scores
            eligible &= self.items["avg_rating"].iloc[: self.n_items].fillna(0).to_numpy() >= 3.7
        scores[~eligible] = -np.inf
        top = self._top(scores, set(exclude or ()) | set(history), k)
        return [self._rec(i, scores[i], self.explain(int(i), history)) for i in top]

    def onboarding_seeds(self, n: int = 24, exclude: set[int] | None = None) -> list[Recommendation]:
        seen = set(exclude or ())
        return [self._rec(i, 0.0, "Tell us what you think") for i in self.seeds if i not in seen][:n]

    @one_at_a_time
    def lab_compare(
        self, history: list[int], k: int = 10, exclude: set[int] | None = None, lam: float = 1.0
    ) -> dict[str, list[Recommendation]]:
        """Model Lab: every model's top-k for the same history, side by side (``lam`` < 1 applies MMR
        diversity to the hybrid, as the diversity slider does)."""
        return {
            m: self.recommend(history, exclude, k=k, model=m, lam=lam if m == "hybrid" else 1.0)
            for m in self.models
        }
