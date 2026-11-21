"""Parsers for MovieLens-1M (``::``-separated, Latin-1 .dat files) and MovieLens-32M (UTF-8 CSV files).

ML-1M is the research benchmark. ML-32M (ratings up to October 2023) trains the models behind the live
app: a recent time window, k-core filtered, so the catalogue holds the films people actually watch now.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.schema import INTERACTION_COLUMNS, INTERACTION_DTYPES, RecDataset

ENCODING = "latin-1"

# Code tables copied from the ML-1M README.
AGE_GROUPS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
OCCUPATIONS = {
    0: "other/not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
# "Misérables, Les" -> "Les Misérables"; "Slingshot, The (Kådisbellan )" keeps the alt title.
_ARTICLE_RE = re.compile(
    r"^(?P<base>.+?), (?P<article>The|A|An|Les|La|Le|L'|Il|El|Das|Die|Der|Det|Den|Lo|Los|Las|Un|Une)"
    r"(?P<rest>\s*\(.*\))?$"
)


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    # "::" is a multi-char separator, which forces pandas' slow Python engine. Rewriting it to
    # a tab first lets the fast C parser read 1M rows in about a second.
    text = path.read_text(encoding=ENCODING).replace("::", "\t")
    return pd.read_csv(
        io.StringIO(text), sep="\t", header=None, names=names, quoting=3, dtype=str, keep_default_na=False
    )


def split_title(raw: str) -> tuple[str, int | None]:
    """'Toy Story (1995)' -> ('Toy Story', 1995). Also moves trailing articles to the front."""
    match = _YEAR_RE.search(raw)
    year = int(match.group(1)) if match else None
    title = raw[: match.start()] if match else raw.strip()
    art = _ARTICLE_RE.match(title)
    if art:
        sep = "" if art["article"].endswith("'") else " "
        title = f"{art['article']}{sep}{art['base']}{art['rest'] or ''}"
    return title.strip(), year


def load_ml1m(directory: Path) -> RecDataset:
    ratings = _read_dat(directory / "ratings.dat", INTERACTION_COLUMNS)
    ratings = ratings.astype(INTERACTION_DTYPES)

    movies = _read_dat(directory / "movies.dat", ["item_id", "raw_title", "genres"])
    parsed = movies["raw_title"].map(split_title)
    items = pd.DataFrame(
        {
            "item_id": movies["item_id"].astype("int32"),
            "title": parsed.map(lambda t: t[0]),
            "year": pd.array(parsed.map(lambda t: t[1]).tolist(), dtype="Int16"),
            "genres": movies["genres"].map(lambda g: g.split("|") if g else []),
            "raw_title": movies["raw_title"],
        }
    )

    users_raw = _read_dat(directory / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    age = users_raw["age"].astype(int)
    occupation = users_raw["occupation"].astype(int)
    users = pd.DataFrame(
        {
            "user_id": users_raw["user_id"].astype("int32"),
            "gender": users_raw["gender"],
            "age_code": age.astype("int8"),
            "age_group": age.map(AGE_GROUPS),
            "occupation_code": occupation.astype("int8"),
            "occupation": occupation.map(OCCUPATIONS),
            "zip": users_raw["zip"],
        }
    )

    if users["age_group"].isna().any() or users["occupation"].isna().any():
        raise ValueError("Unknown age or occupation code in users.dat")
    unknown_genres = set(items["genres"].explode().dropna()) - set(GENRES)
    if unknown_genres:
        raise ValueError(f"Unknown genres in movies.dat: {sorted(unknown_genres)}")

    return RecDataset(
        name="ml-1m",
        interactions=ratings,
        items=items,
        users=users,
        meta={"rating_scale": [1, 5], "source": "https://grouplens.org/datasets/movielens/1m/"},
    ).validate()


# ML-32M genres, renamed to the ML-1M vocabulary where they differ; "IMAX" is a screen format, not a genre.
ML32M_GENRE_MAP = {"Children": "Children's"}
ML32M_DROP_GENRES = {"IMAX", "(no genres listed)"}


def _kcore(liked: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Repeatedly drop films with < min_item likes and users with < min_user likes until stable."""
    while True:
        kept = liked[liked.groupby("item_id")["user_id"].transform("size") >= min_item]
        kept = kept[kept.groupby("user_id")["item_id"].transform("size") >= min_user]
        if len(kept) == len(liked):
            return kept
        liked = kept


def ml32m_genres(raw: str) -> list[str]:
    out = []
    for g in raw.split("|"):
        if g and g not in ML32M_DROP_GENRES:
            out.append(ML32M_GENRE_MAP.get(g, g))
    return out


APP_USER_OFFSET = 10_000_000  # app users get ids far above MovieLens's (max 200,948)
APP_COLUMNS = ["user_key", "movie_id", "rating", "timestamp"]


def app_ratings(path: Path | None = None) -> pd.DataFrame | None:
    """The live app's ratings, exported by its API (`GET /ops/ratings.csv`) with pseudonymous user keys,
    as MovieLens-style rows; None when nothing was imported. Each key gets a stable integer id."""
    from recsys.paths import raw_dir

    path = path or raw_dir("app") / "ratings.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path, dtype={"user_key": str})
    if list(raw.columns) != APP_COLUMNS:
        raise ValueError(f"{path}: expected columns {APP_COLUMNS}, got {list(raw.columns)}")
    keys = sorted(raw["user_key"].unique())
    ids = {k: APP_USER_OFFSET + i for i, k in enumerate(keys)}
    return pd.DataFrame(
        {
            "user_id": raw["user_key"].map(ids).astype("int32"),
            "item_id": raw["movie_id"].astype("int32"),
            "rating": raw["rating"].astype("float32"),
            "timestamp": raw["timestamp"].astype("int64"),
        }
    )


def load_ml32m(
    directory: Path,
    *,
    since: str = "2018-01-01",
    positive_threshold: float = 4.0,
    min_user: int = 5,
    min_item: int = 20,
    app_file: Path | None = None,
    new_films: pd.DataFrame | None = None,
) -> RecDataset:
    """MovieLens-32M, restricted to ratings made on or after ``since``, plus the live app's own ratings
    when they have been imported (``app_file``, default data/raw/app/ratings.csv).

    Users keep every rating they made in the window, provided they liked (rating >= threshold) at least
    ``min_user`` of the kept films; films are kept when at least ``min_item`` of those users liked them
    (iterated to a fixed point, as for Amazon). Original MovieLens ids are kept, so TMDB links and tags
    attach directly.

    A new film (one released after MovieLens's data, `recsys newfilms`) that enough app users liked to pass
    the same filter "graduates": it joins the catalogue with its title, year and genres from the new-film
    tables (``new_films``, by default wikipedia.new_film_catalogue()), and every model then learns it.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    types = {"userId": pa.int32(), "movieId": pa.int32(), "rating": pa.float32(), "timestamp": pa.int64()}
    table = pacsv.read_csv(
        directory / "ratings.csv", convert_options=pacsv.ConvertOptions(column_types=types)
    )
    start = int(datetime.fromisoformat(since).replace(tzinfo=UTC).timestamp())
    ts = table["timestamp"].to_numpy()
    window = ts >= start
    ratings = pd.DataFrame(
        {
            "user_id": table["userId"].to_numpy()[window],
            "item_id": table["movieId"].to_numpy()[window],
            "rating": table["rating"].to_numpy()[window],
            "timestamp": ts[window],
        }
    )
    del table, ts, window
    app = app_ratings(app_file)
    if app is not None:  # the live app's own users (recsys import-app-ratings), same window and filters
        app = app[app["timestamp"] >= start]
        ratings = pd.concat([ratings, app], ignore_index=True)
    liked = ratings.loc[ratings["rating"] >= positive_threshold, ["user_id", "item_id"]]
    core = _kcore(liked, min_user, min_item)
    users = np.unique(core["user_id"].to_numpy())
    films = np.unique(core["item_id"].to_numpy())
    ratings = ratings[ratings["user_id"].isin(users) & ratings["item_id"].isin(films)].reset_index(drop=True)
    ratings = ratings.astype(INTERACTION_DTYPES)

    movies = pd.read_csv(directory / "movies.csv", dtype={"movieId": "int32", "title": str, "genres": str})
    movies = movies[movies["movieId"].isin(films)]
    parsed = movies["title"].map(split_title)
    items = pd.DataFrame(
        {
            "item_id": movies["movieId"].astype("int32").to_numpy(),
            "title": parsed.map(lambda t: t[0]).to_numpy(),
            "year": pd.array(parsed.map(lambda t: t[1]).tolist(), dtype="Int16"),
            "genres": movies["genres"].fillna("").map(ml32m_genres).to_numpy(),
            "raw_title": movies["title"].to_numpy(),
        }
    )
    extra = sorted(set(films.tolist()) - set(items["item_id"].tolist()))
    graduated: list[int] = []
    if extra:  # liked in the app, unknown to MovieLens: new films
        from recsys.data.wikipedia import new_film_catalogue

        catalogue = (new_films if new_films is not None else new_film_catalogue()).set_index("movie_id")
        graduated = [f for f in extra if f in catalogue.index]
        rows = catalogue.loc[graduated]
        items = pd.concat(
            [
                items,
                pd.DataFrame(
                    {
                        "item_id": np.asarray(graduated, dtype="int32"),
                        "title": rows["title"].to_numpy(),
                        "year": pd.array(rows["year"].tolist(), dtype="Int16"),
                        "genres": [list(g) for g in rows["genres"]],
                        "raw_title": [f"{t} ({y})" for t, y in zip(rows["title"], rows["year"], strict=True)],
                    }
                ),
            ],
            ignore_index=True,
        )
        unknown = set(extra) - set(graduated)
        if unknown:  # neither a MovieLens film nor a known new film: nothing to describe it with
            ratings = ratings[~ratings["item_id"].isin(unknown)].reset_index(drop=True)
    return RecDataset(
        name="ml-32m",
        interactions=ratings,
        items=items,
        users=None,
        meta={
            "rating_scale": [0.5, 5],
            "source": "https://grouplens.org/datasets/movielens/32m/",
            "window_since": since,
            "app_ratings_imported": 0 if app is None else len(app),
            "new_films_graduated": len(graduated),
            "kcore": {
                "min_user_likes": min_user,
                "min_film_likes": min_item,
                "liked_threshold": positive_threshold,
            },
        },
    ).validate()
