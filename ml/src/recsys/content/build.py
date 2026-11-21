"""Build each movie's text and sentence embedding, and add them to a serving bundle.

`recsys content <dataset>` writes data/processed/<dataset>/content.parquet (keywords + text) and
text_embeddings.npy; `recsys export` (or `recsys content --bundle`) then copies the vectors, the
keywords and the encoder files into the bundle, so the API can embed a search query the same way.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.content.encoder import MINILM, TextEncoder, fetch_text_model
from recsys.content.keywords import load_keywords
from recsys.paths import processed_dir

log = logging.getLogger(__name__)

ITEM_MAX_TOKENS = 256  # item texts (genres, keywords, plot); queries are short


def item_text(
    title: str,
    year: int | None,
    genres: list[str],
    keywords: list[str],
    overview: str | None = None,
    tagline: str | None = None,
) -> str:
    """One movie as a sentence about what it is: year, genres, viewer keywords, then the plot. Keywords come
    before the plot so truncation never cuts them. The title is left out: in a sentence embedding its words
    outweighed the description, so "time travel" found "A Brief History of Time" first, and titles have
    their own search. Only a film with nothing else to go on is described by its title."""
    described = bool(genres or keywords or any(isinstance(x, str) and x.strip() for x in (overview, tagline)))
    parts = (
        [f"A {year} film." if year else "A film."]
        if described
        else [f"{title} ({year})." if year else f"{title}."]
    )
    if genres:
        parts.append("Genres: " + ", ".join(genres) + ".")
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords) + ".")
    for extra in (tagline, overview):
        if isinstance(extra, str) and extra.strip():
            parts.append(extra.strip())
    return " ".join(parts)


PLOT_CHARS = 700  # the opening of a summary: premise and kind of film, before production details


def plot_text(title: str, overview: str | None, tagline: str | None = None) -> str | None:
    """A film's story for its plot vector: tagline and summary, with the film's own title replaced (its
    words would match a query by name, not by meaning: "slow-burn" found "Burning"), cut at a sentence end
    after about 700 characters."""
    text = " ".join(x.strip() for x in (tagline, overview) if isinstance(x, str) and x.strip())
    if not text:
        return None
    names = {title, re.sub(r"\s*\([^()]*\)\s*$", "", title)}  # "Spirited Away (Sen to ...)" too
    for name in sorted((n for n in names if len(n) >= 2), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "the film", text, flags=re.IGNORECASE)
    if len(text) > PLOT_CHARS:
        cut = text[:PLOT_CHARS]
        text = cut[: cut.rfind(". ") + 1] or cut
    return text


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def film_vectors(encoder, films: list[tuple]) -> np.ndarray:
    """One unit vector per film from (title, year, genres, keywords, overview, tagline). The tags text
    (year, genres, viewer keywords) and the plot text are embedded separately and averaged (late fusion):
    one long text diluted both in MiniLM's single vector and let title words through."""
    tags = [item_text(t, y, g, k) for t, y, g, k, _, _ in films]
    vectors = _unit(np.asarray(encoder.encode(tags), dtype=np.float64))
    plots = [plot_text(t, o, tl) for t, _, _, _, o, tl in films]
    with_plot = [i for i, p in enumerate(plots) if p]
    if with_plot:
        plot_vectors = _unit(np.asarray(encoder.encode([plots[i] for i in with_plot]), dtype=np.float64))
        vectors[with_plot] += plot_vectors
    return _unit(vectors).astype(np.float32)


def _optional(path: Path, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path).drop_duplicates("item_id").set_index("item_id")
    keep = [c for c in cols if c in df]
    return df[keep] if keep else None


def _wiki(src: Path, cols: list[str]) -> pd.DataFrame | None:
    """wiki.parquet's ``cols`` by item_id (`recsys wiki`), plus the same for new films that graduated into
    the catalogue (liked by enough app users; their poster and summary come from the new-film tables)."""
    from recsys.data.wikipedia import NEW_ID_OFFSET, new_film_catalogue

    wiki = _optional(src / "wiki.parquet", cols)
    if not (src / "items.parquet").exists():
        return wiki
    items = pd.read_parquet(src / "items.parquet", columns=["item_id"])
    graduated = items["item_id"][items["item_id"] >= NEW_ID_OFFSET]
    if graduated.empty:
        return wiki
    films = new_film_catalogue().set_index("movie_id")
    extra = films.reindex(graduated.to_numpy())[[c for c in cols if c in films]].dropna(how="all")
    extra.index.name = "item_id"
    return extra if wiki is None else pd.concat([wiki, extra[~extra.index.isin(wiki.index)]])


def build_content(dataset: str, encoder: TextEncoder | None = None) -> pd.DataFrame:
    """Keywords, text and embedding for every movie of ``dataset``; saved next to its processed data."""
    out_dir = processed_dir(dataset)
    items = pd.read_parquet(out_dir / "items.parquet")
    keywords = load_keywords(items["item_id"])
    tmdb = _optional(out_dir / "tmdb.parquet", ["overview", "tagline"])
    wiki = _wiki(out_dir, ["summary"])  # `recsys wiki`: when there is no TMDB plot
    rows, films = [], []
    for item in items.itertuples(index=False):
        meta = dict(tmdb.loc[item.item_id]) if tmdb is not None and item.item_id in tmdb.index else {}
        if not isinstance(meta.get("overview"), str) and wiki is not None and item.item_id in wiki.index:
            meta["overview"] = wiki.loc[item.item_id, "summary"]
        year = None if pd.isna(getattr(item, "year", None)) else int(item.year)
        genres = getattr(item, "genres", None)
        film = (
            item.title,
            year,
            [] if genres is None else list(genres),
            keywords.get(int(item.item_id), []),
            meta.get("overview"),
            meta.get("tagline"),
        )
        films.append(film)
        rows.append(
            {
                "item_id": int(item.item_id),
                "keywords": film[3],
                "text": item_text(*film[:4]),
                "plot": plot_text(film[0], film[4], film[5]),
            }
        )
    content = pd.DataFrame(rows)
    encoder = encoder or TextEncoder(fetch_text_model(), max_length=ITEM_MAX_TOKENS)
    vectors = film_vectors(encoder, films)
    content.to_parquet(out_dir / "content.parquet", index=False)
    np.save(out_dir / "text_embeddings.npy", vectors)
    covered = (content["keywords"].map(len) > 0).mean()
    log.info("%d movies embedded; %.1f%% have viewer keywords", len(content), 100 * covered)
    return content


def add_content_to_bundle(bundle: Path, dataset: str) -> dict:
    """Copy keywords, vectors and the encoder into an exported bundle (aligned to its item order)."""
    bundle = Path(bundle)
    src = processed_dir(dataset)
    if not (src / "content.parquet").exists():
        raise FileNotFoundError(f"no content for {dataset}: run `recsys content {dataset}` first")
    content = pd.read_parquet(src / "content.parquet")
    vectors = np.load(src / "text_embeddings.npy")
    row_of = {int(m): r for r, m in enumerate(content["item_id"])}

    catalog = pd.read_parquet(bundle / "items.parquet")
    rows = np.array([row_of.get(int(m), -1) for m in catalog["movie_id"]])
    aligned = np.zeros((len(catalog), vectors.shape[1]), dtype=np.float32)
    aligned[rows >= 0] = vectors[rows[rows >= 0]]
    np.save(bundle / "text_embeddings.npy", aligned)

    keywords = content.set_index("item_id")["keywords"]
    catalog["keywords"] = [list(keywords.get(int(m), [])) for m in catalog["movie_id"]]
    # Posters and summaries from Wikipedia (`recsys wiki`): every film gets its artwork; a TMDB plot, when
    # fetched, stays the summary of choice
    wiki = _wiki(src, ["poster_url", "summary"])
    if wiki is not None:
        catalog["poster"] = catalog["movie_id"].map(wiki["poster_url"])
        summary = catalog["movie_id"].map(wiki["summary"])
        # a TMDB plot, from TMDB's own table (not the bundle's overview column, which an earlier run of
        # this function may have filled with Wikipedia's summary)
        tmdb = _optional(src / "tmdb.parquet", ["overview"])
        plot = (
            catalog["movie_id"].map(tmdb["overview"])
            if tmdb is not None and "overview" in tmdb
            else pd.Series(None, index=catalog.index, dtype=object)
        )
        had = plot.notna()
        catalog["overview"] = plot.where(had, summary)
        catalog["overview_source"] = np.where(had, "tmdb", np.where(summary.notna(), "wikipedia", None))
    # Display/quality statistics for the app (average rating, number of ratings) over all ratings. They
    # never reach the evaluated models; the app uses them on movie pages and to filter "hidden gems".
    inter_path = src / "interactions.parquet"
    if inter_path.exists():
        stats = pd.read_parquet(inter_path, columns=["item_id", "rating"]).groupby("item_id")["rating"]
        catalog["avg_rating"] = catalog["movie_id"].map(stats.mean()).astype("float32")
        catalog["n_ratings"] = catalog["movie_id"].map(stats.size()).fillna(0).astype("int32")
    links = _optional(src / "links.parquet", ["tmdb_id", "imdb_id"])
    if links is not None:
        missing = [c for c in links.columns if c not in catalog]
        if missing:
            catalog = catalog.join(links[missing], on="movie_id")
    catalog.to_parquet(bundle / "items.parquet", index=False)

    model_dir = fetch_text_model()
    target = bundle / "text_model"
    target.mkdir(exist_ok=True)
    for name in ("model.onnx", "tokenizer.json"):
        shutil.copyfile(model_dir / name, target / name)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["text_model"] = {
        "name": MINILM.name,
        "repo": MINILM.repo,
        "revision": MINILM.revision,
        "dim": MINILM.dim,
        "item_max_tokens": ITEM_MAX_TOKENS,
        "items_with_keywords": int(sum(len(k) > 0 for k in catalog["keywords"])),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["text_model"]
