"""New films that no viewer has rated yet (recent Telugu, Tamil, Kannada and Malayalam releases),
recommended from their story.

Collaborative models cannot score a film without ratings: the item cold-start problem. Following the idea
of predicting a model's latent factors from content (van den Oord et al., 2013, for music from audio), a
ridge regression maps each film's sentence embedding (MiniLM over its year, genres, keywords and summary)
to its iALS item factors, learned on the catalogue's films. A new film's predicted factors then meet the
user's iALS vector (the exact fold-in from their history) just as a known film's factors do.

`evaluate_mapping` checks this on catalogue films treated as new; `build_new_films` adds new films to a
bundle (new_items.parquet, their predicted factors and text vectors), which the engine serves as one row.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.wikipedia import NEW_ID_OFFSET, describes_film
from recsys.serving.engine import Engine

log = logging.getLogger(__name__)
K = 10
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def fit_mapping(X: np.ndarray, Q: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression from text vectors X (n, d) to factors Q (n, k), with a bias row: W is (d + 1, k)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0  # the bias is not shrunk
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Q)


def predict_factors(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ W[:-1] + W[-1]


def choose_alpha(X: np.ndarray, Q: np.ndarray, seed: int = 0) -> tuple[float, dict[float, float]]:
    """The ridge strength with the best R² on a held-out fifth of the training films."""
    order = np.random.default_rng(seed).permutation(len(X))
    cut = len(X) // 5
    val, fit = order[:cut], order[cut:]
    r2 = {}
    for a in ALPHAS:
        pred = predict_factors(fit_mapping(X[fit], Q[fit], a), X[val])
        r2[a] = 1 - ((Q[val] - pred) ** 2).sum() / ((Q[val] - Q[val].mean(0)) ** 2).sum()
    return max(r2, key=r2.get), r2


def fold_in(Q: np.ndarray, items: np.ndarray, alpha: float, gram: np.ndarray) -> np.ndarray:
    """iALS's exact user vector for the liked ``items`` (the same half-step as IALSScorer)."""
    if not len(items):
        return np.zeros(Q.shape[1])
    Qu = Q[items]
    return np.linalg.solve(gram + alpha * (Qu.T @ Qu), (1.0 + alpha) * Qu.sum(axis=0))


def _ials(engine: Engine):
    scorer = engine.scorers.get("ials")
    if scorer is None:
        raise ValueError("the bundle has no iALS generator: new films are placed in its factor space")
    return scorer


def _hidden_setup(engine: Engine, seed: int, holdout: float) -> dict:
    """A random ``holdout`` share of the catalogue's films treated as new (no ratings, no factors), the
    text-to-factors mapping learned on the rest, and what both evaluations need about them."""
    ials = _ials(engine)
    Q = ials.Q
    text = engine.text_emb[: engine.n_items].astype(np.float64)
    has_text = np.any(text, axis=1)
    rng = np.random.default_rng(seed)
    hidden = np.sort(rng.choice(np.flatnonzero(has_text), size=int(holdout * has_text.sum()), replace=False))
    known = np.setdiff1d(np.flatnonzero(has_text), hidden)
    alpha, r2 = choose_alpha(text[known], Q[known])
    W = fit_mapping(text[known], Q[known], alpha)
    log.info(
        "mapping: alpha %s (R² on held-out training films %s)", alpha, {k: round(v, 3) for k, v in r2.items()}
    )
    is_hidden = np.zeros(engine.n_items, dtype=bool)
    is_hidden[hidden] = True
    return {
        "ials": ials,
        "Q": Q,
        "text": text,
        "rng": rng,
        "hidden": hidden,
        "alpha": alpha,
        "predicted": predict_factors(W, text[hidden]),
        "is_hidden": is_hidden,
        "col": {int(i): j for j, i in enumerate(hidden)},
        "gram": Q[~is_hidden].T @ Q[~is_hidden] + ials.reg * np.eye(Q.shape[1]),
    }


def evaluate_mapping(
    engine: Engine, test_targets, seed: int = 42, holdout: float = 0.1, max_users: int = 5000
) -> pd.DataFrame:
    """Hide a random 10% of the catalogue's films as if they were new: no ratings, no factors. Learn the
    mapping on the rest, and for test users who went on to like a hidden film, rank only the hidden films:
    NDCG@10 and hit@10 against what they liked next. User vectors come from their history without the
    hidden films. Compared with a random order, content alone (cosine between the user's liked films' text
    and the film's), and the hidden films' true iALS factors (the upper bound the mapping aims at)."""
    h = _hidden_setup(engine, seed, holdout)
    ials, Q, text, rng, hidden = h["ials"], h["Q"], h["text"], h["rng"], h["hidden"]
    predicted, is_hidden, col, gram = h["predicted"], h["is_hidden"], h["col"], h["gram"]
    users = engine._dataset_users
    targets = test_targets.tocsr()
    discount = 1 / np.log2(np.arange(2, K + 2))
    norm_text = text[hidden] / np.maximum(np.linalg.norm(text[hidden], axis=1, keepdims=True), 1e-9)
    results: dict[str, list[tuple[float, float]]] = {}
    candidates = rng.permutation(len(users["user_ids"]))
    n = 0
    for pos in candidates:
        liked_next = [
            col[i] for i in targets.indices[targets.indptr[pos] : targets.indptr[pos + 1]] if i in col
        ]
        history = users["items"][users["indptr"][pos] : users["indptr"][pos + 1]]
        history = history[~is_hidden[history]]
        if not liked_next or not len(history):
            continue
        u = fold_in(Q, history, ials.alpha, gram)
        profile = text[history].mean(axis=0)
        profile /= max(np.linalg.norm(profile), 1e-9)
        scores = {
            "random": rng.random(len(hidden)),
            "content only (text cosine)": norm_text @ profile,
            "ML: text mapped to iALS factors": predicted @ u,
            "upper bound: true iALS factors": Q[hidden] @ u,
        }
        relevant = set(liked_next)
        ideal = discount[: min(K, len(relevant))].sum()
        for name, s in scores.items():
            hits = np.array([j in relevant for j in np.argsort(-s)[:K]], dtype=np.float64)
            results.setdefault(name, []).append((float(hits.any()), float((hits * discount).sum() / ideal)))
        n += 1
        if n >= max_users:
            break
    rows = [
        {
            "method": name,
            "hit@10": np.mean([hit for hit, _ in v]),
            "NDCG@10": np.mean([d for _, d in v]),
            "users": len(v),
        }
        for name, v in results.items()
    ]
    table = pd.DataFrame(rows)
    table.attrs.update({"hidden films": len(hidden), "alpha": h["alpha"]})
    return table


def warm_factors(
    prior: np.ndarray, users: np.ndarray, liked: np.ndarray, confidence: float, strength: float
) -> np.ndarray:
    """A new film's iALS factors from the guess its story gives and the viewers who have rated it so far:
    collaborative topic regression (Wang & Blei, 2011). The factors get a Gaussian prior centred on the
    text's prediction; each rating is an iALS observation with the model's confidence. The MAP estimate:

        q = (c * sum_u p_u p_u^T + strength * I)^-1 (c * sum_u r_u p_u + strength * prior)

    ``users``: (n, k) the raters' iALS vectors; ``liked``: (n,) 1 for a like, 0 for a low rating. With no
    ratings q is the prior; with many it is what the raters' tastes say, as for a film the model trained
    on. ``strength`` sets how many ratings it takes to move away from the story's guess."""
    if not len(users):
        return np.asarray(prior, dtype=np.float64).copy()
    users = np.asarray(users, dtype=np.float64)
    A = confidence * users.T @ users + strength * np.eye(len(prior))
    b = confidence * (np.asarray(liked, dtype=np.float64) @ users) + strength * np.asarray(prior)
    return np.linalg.solve(A, b)


WARM_LIKES = (0, 1, 3, 10, 30)
STRENGTHS = (1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_STRENGTH = 10.0  # until `recsys newfilms --evaluate` has chosen one on validation data


def _rank_metrics(U: np.ndarray, items: np.ndarray, relevant: list[list[int]]) -> tuple[float, float]:
    """Mean hit@10 and NDCG@10 of ranking ``items`` (n, k) for users ``U`` (m, k)."""
    S = U @ items.T
    top = np.argpartition(-S, K - 1, axis=1)[:, :K]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, axis=1), axis=1), axis=1)
    discount = 1 / np.log2(np.arange(2, K + 2))
    hits, ndcgs = [], []
    for row, rel in zip(top, relevant, strict=True):
        found = np.isin(row, rel).astype(np.float64)
        hits.append(found.any())
        ndcgs.append((found * discount).sum() / discount[: min(K, len(rel))].sum())
    return float(np.mean(hits)), float(np.mean(ndcgs))


def evaluate_warmup(
    engine: Engine,
    val_targets,
    test_targets,
    seed: int = 42,
    holdout: float = 0.1,
    likes: tuple[int, ...] = WARM_LIKES,
    strengths: tuple[float, ...] = STRENGTHS,
    max_users: int = 5000,
) -> pd.DataFrame:
    """How a new film's recommendations improve as people rate it (warm_factors). The same films are
    treated as new as in evaluate_mapping. The dataset's viewers are split in two halves:
    - raters: for each film treated as new, the first n of its likers in this half (in random order) "rate"
      it, and the film's factors are re-estimated from the story's guess and those n likes;
    - viewers: as in evaluate_mapping, those who went on to like a film treated as new get only those films
      ranked, from their history without them.
    The prior strength is chosen on the validation targets (best mean NDCG@10 over 1, 3 and 10 likes),
    then every n is measured on the test targets, next to the films' true factors (the upper bound)."""
    h = _hidden_setup(engine, seed, holdout)
    ials, Q, hidden, predicted = h["ials"], h["Q"], h["hidden"], h["predicted"]
    is_hidden, col, gram = h["is_hidden"], h["col"], h["gram"]
    users = engine._dataset_users
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(users["user_ids"]))
    raters, viewers = order[: len(order) // 2], order[len(order) // 2 :]

    def split_history(pos: int) -> tuple[np.ndarray, np.ndarray]:
        hist = users["items"][users["indptr"][pos] : users["indptr"][pos + 1]]
        return hist[~is_hidden[hist]], hist[is_hidden[hist]]

    likers: list[list[np.ndarray]] = [[] for _ in hidden]
    for pos in raters:
        seen, liked_hidden = split_history(pos)
        if not len(seen) or not len(liked_hidden):
            continue
        p = fold_in(Q, seen, ials.alpha, gram)
        for i in liked_hidden:
            likers[col[int(i)]].append(p)
    for lst in likers:
        rng.shuffle(lst)

    def audience(targets) -> tuple[np.ndarray, list[list[int]]]:
        targets = targets.tocsr()
        U, rel = [], []
        for pos in viewers:
            nxt = [col[i] for i in targets.indices[targets.indptr[pos] : targets.indptr[pos + 1]] if i in col]
            seen, _ = split_history(pos)
            if nxt and len(seen):
                U.append(fold_in(Q, seen, ials.alpha, gram))
                rel.append(nxt)
            if len(U) >= max_users:
                break
        return np.array(U), rel

    def warmed(n: int, strength: float) -> tuple[np.ndarray, float]:
        out = predicted.copy()
        used = []
        for j, lst in enumerate(likers):
            chosen = lst[:n]
            used.append(len(chosen))
            if chosen:
                out[j] = warm_factors(
                    predicted[j], np.array(chosen), np.ones(len(chosen)), 1 + ials.alpha, strength
                )
        return out, float(np.mean(used))

    U_val, rel_val = audience(val_targets)
    choice = {
        s: float(np.mean([_rank_metrics(U_val, warmed(n, s)[0], rel_val)[1] for n in (1, 3, 10)]))
        for s in strengths
    }
    strength = max(choice, key=choice.get)
    log.info(
        "warm-up prior strength %s (validation NDCG@10 %s)",
        strength,
        {k: round(v, 4) for k, v in choice.items()},
    )

    U, rel = audience(test_targets)
    rows = []
    for n in likes:
        factors, used = warmed(n, strength)
        hit, ndcg = _rank_metrics(U, factors, rel)
        rows.append(
            {"likes": str(n), "hit@10": hit, "NDCG@10": ndcg, "likes per film (mean)": used, "users": len(U)}
        )
    hit, ndcg = _rank_metrics(U, Q[hidden], rel)
    rows.append(
        {
            "likes": "all (true factors)",
            "hit@10": hit,
            "NDCG@10": ndcg,
            "likes per film (mean)": None,
            "users": len(U),
        }
    )
    table = pd.DataFrame(rows)
    table.attrs.update({"prior": strength, "hidden films": len(hidden), "validation": choice})
    return table


def known_imdb_ids(engine: Engine) -> set[int]:
    """IMDb numbers of the films the models know. Not those of new films already in the bundle: a rebuild
    replaces them, so they must not be skipped as if the catalogue had them."""
    imdb = engine.items.get("imdb_id", pd.Series(dtype=float)).iloc[: engine.n_items]
    return {int(x) for x in pd.to_numeric(imdb, errors="coerce").dropna()}


def _key(text: str) -> str:
    """Letters and digits only, accents dropped, lower case: to compare titles as written in an article."""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _rows(films: pd.DataFrame, keep) -> pd.DataFrame:
    """The rows where ``keep`` (one bool per row) is true. Not ``films[list]``: with no rows the list is
    empty, and pandas reads an empty list as a selection of no *columns*."""
    return films.loc[np.asarray(list(keep), dtype=bool)]


def _app_id(wikidata: str) -> int:
    return NEW_ID_OFFSET + int(wikidata[1:])


def usable_films(films: pd.DataFrame, known_imdb: set[int]) -> pd.DataFrame:
    """The fetched films worth adding: not already in the catalogue (by IMDb id), with a summary that is
    about this film (wikipedia.describes_film: not a person, a list or an older film the article redirects
    to), each once (a bilingual release listed under two languages: the first; two Wikidata items for one
    article, a working title or a dubbed version: the one the article names)."""
    films = films.copy()
    # Wikidata's IMDb field sometimes holds another kind of IMDb id (a person's, "nm..."): titles only
    films["imdb"] = films["imdb"].where(films["imdb"].str.fullmatch(r"tt\d+", na=False))
    films["imdb_num"] = pd.to_numeric(films["imdb"].str.removeprefix("tt"), errors="coerce")
    films = films[~films["imdb_num"].isin(known_imdb) & films["summary"].notna()]
    films = _rows(films, (describes_film(s, y) for s, y in zip(films["summary"], films["year"], strict=True)))
    films = films.drop_duplicates("wikidata")
    named = [_key(t)[:10] in _key(s[:200]) for t, s in zip(films["title"], films["summary"], strict=True)]
    films = films.assign(named=named).sort_values("named", ascending=False, kind="stable")
    films = films.drop_duplicates("summary").sort_index().drop(columns="named")
    return films.reset_index(drop=True)


def film_rows(films: pd.DataFrame) -> pd.DataFrame:
    """Bundle rows (new_items.parquet) for usable films."""
    links = films["links"] if "links" in films else pd.Series(0, index=films.index)
    return pd.DataFrame(
        {
            "movie_id": [NEW_ID_OFFSET + int(q[1:]) for q in films["wikidata"]],
            "title": films["title"].to_numpy(),
            "year": pd.array(films["year"].to_numpy(), dtype="Int16"),
            "genres": [list(g) for g in films["genres"]],
            "popularity": 0,
            # as the catalogue stores it: digits as text
            "imdb_id": films["imdb"].str.removeprefix("tt").to_numpy(),
            "poster": films["poster_url"].to_numpy(),
            "overview": films["summary"].to_numpy(),
            "overview_source": "wikipedia",
            "language": films["language"].to_numpy(),
            # Wikipedia language editions covering the film: how widely it is known (new users' ranking)
            "links": pd.to_numeric(links, errors="coerce").fillna(0).astype("int32").to_numpy(),
            "keywords": [[] for _ in range(len(films))],
        }
    )


def item_encoder(engine: Engine):
    """The sentence encoder for film texts, set up exactly as for the catalogue's films (so the mapping
    learned on those applies), from the bundle's copy of the model."""
    from recsys.content.build import ITEM_MAX_TOKENS
    from recsys.content.encoder import TextEncoder

    return TextEncoder(engine.bundle / "text_model", max_length=ITEM_MAX_TOKENS)


def embed_films(encoder, films: pd.DataFrame) -> np.ndarray:
    from recsys.content.build import film_vectors

    rows_in = zip(films["title"], films["year"], films["genres"], films["summary"], strict=True)
    return film_vectors(encoder, [(t, int(y), list(g), [], s, None) for t, y, g, s in rows_in])


def text_mapping(engine: Engine) -> tuple[np.ndarray, float, float]:
    """The ridge map from a film's text vector to its iALS factors, fitted on all the catalogue's films:
    (W, ridge strength, R² on held-out training films)."""
    ials = _ials(engine)
    text = engine.text_emb[: engine.n_items].astype(np.float64)
    has_text = np.any(text, axis=1)
    alpha, r2 = choose_alpha(text[has_text], ials.Q[has_text])
    return fit_mapping(text[has_text], ials.Q[has_text], alpha), alpha, float(r2[alpha])


def write_new_items(engine: Engine, bundle: Path | None = None) -> None:
    """Save the engine's new films (rows, story-based factors, text vectors) into the bundle, each file
    written whole and then swapped in, so a reader never sees half of one."""
    bundle = Path(bundle or engine.bundle)
    rows = engine.items.iloc[engine.n_items :].drop(columns=["item_idx"]).reset_index(drop=True)

    def swap(name: str, write) -> None:
        tmp = bundle / f"{name}.part"
        write(tmp)
        tmp.replace(bundle / name)

    def save_array(array: np.ndarray):
        def write(path: Path) -> None:
            with open(path, "wb") as f:  # a file object: np.save would add ".npy" to a ".part" path
                np.save(f, array)

        return write

    swap("new_items.parquet", lambda p: rows.to_parquet(p, index=False))
    swap("new_items_factors.npy", save_array(engine.new_prior.astype(np.float32)))
    swap("new_items_text.npy", save_array(engine.text_emb[engine.n_items :]))


def add_releases(engine: Engine, films: pd.DataFrame, encoder=None, mapping=None) -> pd.DataFrame:
    """Add to a running engine the fetched films it has neither in its catalogue nor among its new films
    (the API's daily refresh): their story is embedded and mapped to iALS factors, and they join the new
    films at once. Returns the rows added."""
    known_new = set(engine.items["movie_id"].iloc[engine.n_items :].astype("int64"))
    films = usable_films(films, known_imdb_ids(engine))
    films = _rows(films, (_app_id(q) not in known_new for q in films["wikidata"]))
    films = _rows(films, (_app_id(q) not in engine.movie_to_idx for q in films["wikidata"]))
    if films.empty:
        return film_rows(films)
    W = mapping[0] if mapping is not None else text_mapping(engine)[0]
    vectors = embed_films(encoder or item_encoder(engine), films)
    factors = predict_factors(W, vectors.astype(np.float64))
    rows = film_rows(films)
    engine.add_new_films(rows, factors, vectors)
    return rows


def build_new_films(bundle: Path, films: pd.DataFrame, encoder=None, strength: float | None = None) -> dict:
    """Add ``films`` (from wikipedia.language_films, one or several languages) that the catalogue does not
    have to a bundle, replacing earlier new films: their rows (new_items.parquet), their text vectors and
    their predicted iALS factors (which films: usable_films). ``strength``: the warm-up's prior strength
    (evaluate_warmup); kept from the last build when not given."""
    bundle = Path(bundle)
    engine = Engine(bundle)
    films = usable_films(films, known_imdb_ids(engine))
    # not those that graduated into the catalogue (app/releases.py, movielens.load_ml32m)
    films = _rows(
        films,
        (engine.movie_to_idx.get(_app_id(q), engine.n_items) >= engine.n_items for q in films["wikidata"]),
    )
    films = films.reset_index(drop=True)
    vectors = embed_films(encoder or item_encoder(engine), films)
    W, alpha, r2 = text_mapping(engine)
    factors = predict_factors(W, vectors.astype(np.float64)).astype(np.float32)

    rows = film_rows(films)
    rows.to_parquet(bundle / "new_items.parquet", index=False)
    np.save(bundle / "new_items_factors.npy", factors)
    np.save(bundle / "new_items_text.npy", vectors)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = manifest.get("new_items", {})
    languages = list(dict.fromkeys(films["language"]))  # in the order asked for
    manifest["new_items"] = {
        "films": len(rows),
        "languages": languages,
        "per_language": {lang: int((films["language"] == lang).sum()) for lang in languages},
        "mapping": "ridge: MiniLM text vector -> iALS item factors",
        "alpha": alpha,
        "r2_on_held_out_training_films": round(r2, 4),
        "warmup": "collaborative topic regression: text factors as a Gaussian prior, updated by ratings",
        "prior_strength": float(
            strength if strength is not None else before.get("prior_strength", DEFAULT_STRENGTH)
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["new_items"]
