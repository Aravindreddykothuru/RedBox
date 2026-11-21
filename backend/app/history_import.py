"""Watch history from other services, so Red Box can recommend before the user rates anything.

The user exports their own history and uploads the file. It is read once, in memory, to match titles to the
catalogue, and is never stored: only the films the user then confirms are saved. Supported files, recognised
from their header row:

- Netflix "Viewing activity" (NetflixViewingHistory.csv: Title, Date, newest first), or ViewingActivity.csv
  from Netflix's full data download (Title, Start Time, Duration, Supplemental Video Type). What was
  watched, without ratings. TV episodes ("Show: Season 1: Episode") are skipped; in the full download,
  trailers and films watched for less than 15 minutes in all are skipped too.
- IMDb exports: Const (the IMDb id), Title, Year, Title Type and, for "Your ratings", Your Rating (1-10)
  and Date Rated. Matched by IMDb id. A list without ratings, such as the watchlist, goes to My list.
- Letterboxd's export: the .zip, or ratings.csv, watched.csv, diary.csv or watchlist.csv from it (Date,
  Name, Year, Letterboxd URI and, where rated, Rating in 0.5-5 stars).

A matched film becomes a rating (IMDb or Letterboxd stars, on Red Box's 0.5-5 scale), "watched" (implicit
feedback: counted as liked unless the user rates the film lower) or a My list entry.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import csv
import io
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Literal

import pandas as pd

Kind = Literal["rating", "watched", "watchlist"]
Source = Literal["netflix", "imdb", "letterboxd"]

MAX_BYTES = 10 * 1024 * 1024  # an upload; a zip may unpack to 4x this at most
MIN_NETFLIX_SECONDS = 15 * 60  # full Netflix download: less than this in total is sampling, not watching
LETTERBOXD_FILES = ("ratings.csv", "diary.csv", "watched.csv", "watchlist.csv")
KIND_ORDER = {"rating": 0, "watched": 1, "watchlist": 2}  # what wins when one film appears several ways


class ImportFileError(ValueError):
    """The upload cannot be read; the message is shown to the user."""


@dataclass
class Row:
    """One line of an export, in a common shape."""

    title: str
    kind: Kind
    year: int | None = None
    at: datetime | None = None
    rating: float | None = None  # 0.5-5
    imdb: int | None = None


@dataclass
class Match:
    movie_id: int
    kind: Kind
    source_title: str
    rating: float | None = None
    at: datetime | None = None
    times: int = 1  # how often it was watched or rated (rewatches)


@dataclass
class Preview:
    source: Source
    rows: int  # usable lines read
    matches: list[Match]
    tv_skipped: int = 0
    short_skipped: int = 0
    unmatched: list[str] = field(default_factory=list)  # distinct titles not in the catalogue


# ---------------------------------------------------------------------------------------------- reading


def read_upload(name: str, content: str, encoding: Literal["text", "base64"]) -> list[tuple[str, str]]:
    """(file name, CSV text) pairs: the file itself, or the Letterboxd CSVs inside a .zip."""
    if encoding == "base64":
        try:
            raw = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ImportFileError("the file could not be read") from e
    else:
        raw = content.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ImportFileError("the file is larger than 10 MB")
    if raw[:2] == b"PK":
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as e:
            raise ImportFileError("the zip file is damaged") from e
        with archive:
            files, total = [], 0
            for info in archive.infolist():
                base = info.filename.rsplit("/", 1)[-1].lower()
                if info.is_dir() or base not in LETTERBOXD_FILES:
                    continue
                total += info.file_size
                if total > 4 * MAX_BYTES:
                    raise ImportFileError("the zip unpacks to more than 40 MB")
                files.append((base, archive.read(info).decode("utf-8-sig", errors="replace")))
        if not files:
            raise ImportFileError(
                "no Letterboxd files (ratings.csv, diary.csv, watched.csv, watchlist.csv) in the zip"
            )
        return files
    return [(name.rsplit("/", 1)[-1].lower(), raw.decode("utf-8-sig", errors="replace"))]


def _table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    rows = [{k: (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    return header, rows


def _iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _utc(d: datetime) -> datetime:
    d = d if d.tzinfo else d.replace(tzinfo=UTC)
    return min(d, datetime.now(UTC))  # a clock or export in the future must not reorder the history


def _int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _half_stars(value: float) -> float | None:
    if not 0 < value <= 5:
        return None
    return min(5.0, max(0.5, round(value * 2) / 2))


# Netflix writes dates in the account's locale (12/31/23, 31/12/2023, 2023-12-31, ...). The right reading
# is the one that keeps the file's order, newest first.
_NETFLIX_DATE_FORMATS = (
    "%m/%d/%y",
    "%d/%m/%y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d.%m.%y",
    "%d.%m.%Y",
    "%Y/%m/%d",
)


def _netflix_dates(values: list[str]) -> list[datetime | None]:
    best: list[datetime | None] | None = None
    best_score = -1.0
    for fmt in _NETFLIX_DATE_FORMATS:
        try:
            parsed = [datetime.strptime(v, fmt) if v else None for v in values]
        except ValueError:
            continue
        known = [d for d in parsed if d is not None]
        if not known:
            continue
        in_order = sum(a >= b for a, b in pairwise(known)) / max(1, len(known) - 1)
        if in_order > best_score:
            best, best_score = parsed, in_order
    return [None if d is None else _utc(d) for d in best] if best else [None] * len(values)


def _netflix_rows(header: list[str], rows: list[dict[str, str]], counts: dict[str, int]) -> list[Row]:
    if "Start Time" in header:  # the full data download: one line per viewing session
        timed = "Duration" in header
        seconds: dict[str, int] = defaultdict(int)
        latest: dict[str, datetime | None] = {}
        for r in rows:
            title = r.get("Title", "")
            if r.get("Supplemental Video Type") or not title:
                continue  # trailers, teasers, recaps
            if timed:
                h, m, s = ([int(x) for x in re.findall(r"\d+", r.get("Duration", ""))] + [0, 0, 0])[:3]
                seconds[title] += h * 3600 + m * 60 + s
            at = _iso(r.get("Start Time", ""))
            if title not in latest or (at and (latest[title] is None or at > latest[title])):
                latest[title] = at
        out = []
        for title, at in latest.items():
            if timed and seconds[title] < MIN_NETFLIX_SECONDS and not looks_like_episode(title):
                counts["short"] += 1
                continue
            out.append(Row(title=title, kind="watched", at=at))
        return out
    dates = _netflix_dates([r.get("Date", "") for r in rows])
    return [
        Row(title=r["Title"], kind="watched", at=d)
        for r, d in zip(rows, dates, strict=True)
        if r.get("Title")
    ]


_IMDB_TV = {
    "tv series",
    "tv episode",
    "tv mini series",
    "tvseries",
    "tvepisode",
    "tvminiseries",
    "podcast series",
    "podcast episode",
}


def _imdb_rows(header: list[str], rows: list[dict[str, str]], counts: dict[str, int]) -> list[Row]:
    rating_col = "Your Rating" if "Your Rating" in header else None
    out = []
    for r in rows:
        if r.get("Title Type", "").lower() in _IMDB_TV:
            counts["tv"] += 1
            continue
        tt = re.fullmatch(r"tt0*(\d+)", r.get("Const", ""))
        stars = _int(r.get(rating_col, "")) if rating_col else None
        out.append(
            Row(
                title=r.get("Title", "") or r.get("Original Title", ""),
                kind="rating" if stars else "watchlist",
                year=_int(r.get("Year", "")),
                at=_iso(r.get("Date Rated", "") or r.get("Created", "")),
                rating=_half_stars(stars / 2) if stars else None,
                imdb=int(tt.group(1)) if tt else None,
            )
        )
    return out


def _letterboxd_rows(name: str, rows: list[dict[str, str]]) -> list[Row]:
    out = []
    for r in rows:
        stars = r.get("Rating", "")
        rating = _half_stars(float(stars)) if re.fullmatch(r"\d+(\.\d+)?", stars) else None
        kind: Kind = "rating" if rating else ("watchlist" if "watchlist" in name else "watched")
        out.append(
            Row(
                title=r.get("Name", ""),
                kind=kind,
                year=_int(r.get("Year", "")),
                at=_iso(r.get("Watched Date", "") or r.get("Date", "")),
                rating=rating,
            )
        )
    return out


def parse(files: list[tuple[str, str]]) -> tuple[Source, list[Row], dict[str, int]]:
    """Recognise the export from its header and read every file into rows."""
    source: Source | None = None
    rows: list[Row] = []
    counts = {"tv": 0, "short": 0}
    for name, text in files:
        header, table = _table(text)
        if "Title" in header and ("Date" in header or "Start Time" in header) and "Const" not in header:
            kind: Source = "netflix"
            rows += _netflix_rows(header, table, counts)
        elif "Const" in header and "Title" in header:
            kind = "imdb"
            rows += _imdb_rows(header, table, counts)
        elif "Name" in header and "Letterboxd URI" in header:
            kind = "letterboxd"
            rows += _letterboxd_rows(name, table)
        else:
            raise ImportFileError(
                "this is not a Netflix, IMDb or Letterboxd export: the first line should name its columns "
                "(Netflix: Title, Date; IMDb: Const, Title; Letterboxd: Name, Letterboxd URI)"
            )
        if source not in (None, kind):
            raise ImportFileError("upload one service's export at a time")
        source = kind
    assert source is not None
    return source, rows, counts


# --------------------------------------------------------------------------------------------- matching


_APOSTROPHES = re.compile("['`" + chr(0x2019) + "]")  # straight, backtick and curly: "Schindler's List"


def norm(title: str) -> str:
    """Lower case, no accents or punctuation, "&" as "and", no leading article: "The Lord of the Rings:
    The Fellowship of the Ring" and "Lord of the Rings - Fellowship of the Ring" come out the same."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = (folded if folded.strip() else title).casefold().replace("&", " and ")
    s = _APOSTROPHES.sub("", s)
    s = re.sub(r"[^\w]+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


_TV_PART = re.compile(
    r"^(season|series|volume|vol|part|chapter|episode|ep|book|collection|limited series|miniseries|"
    r"temporada|staffel|saison)\b|^s\d+(e\d+)?$|^e\d+$|^\d+$",
    re.IGNORECASE,
)


def looks_like_episode(title: str) -> bool:
    """Netflix names episodes "Show: Season 1: Episode Title" (and "Show: Limited Series: ...")."""
    parts = [p.strip() for p in title.split(":")]
    return len(parts) >= 3 or (len(parts) == 2 and bool(_TV_PART.match(parts[1])))


_TRAILING_TAG = re.compile(r"\s*\((?:[^()]*)\)\s*$")  # "RRR (Hindi)", "Dune (2021)"
_STOP = {"the", "a", "an", "of", "and", "in", "on", "to", "part", "movie", "film"}


@dataclass
class Catalogue:
    """Lookup tables over the bundle's films: by IMDb id, by normalised title (main title, the full title
    and any alternative title in brackets), and by word for the fuzzy fallback."""

    year: dict[int, int | None]
    popularity: dict[int, float]
    by_imdb: dict[int, int]
    by_title: dict[str, list[int]]
    by_word: dict[str, set[str]]


_CATALOGUES: list[tuple[pd.DataFrame, Catalogue]] = []  # built once per loaded bundle


def catalogue(items: pd.DataFrame) -> Catalogue:
    for known, cat in _CATALOGUES:
        if known is items:
            return cat
    cat = _build_catalogue(items)
    _CATALOGUES[:] = [*_CATALOGUES[-3:], (items, cat)]
    return cat


def _build_catalogue(items: pd.DataFrame) -> Catalogue:
    by_title: dict[str, list[int]] = defaultdict(list)
    by_word: dict[str, set[str]] = defaultdict(set)
    years: dict[int, int | None] = {}
    pops: dict[int, float] = {}
    by_imdb: dict[int, int] = {}
    for row in items.itertuples(index=False):
        mid = int(row.movie_id)
        title = str(row.title)
        year = getattr(row, "year", None)
        years[mid] = None if year is None or pd.isna(year) else int(year)
        pop = getattr(row, "popularity", 0)
        pops[mid] = 0.0 if pop is None or pd.isna(pop) else float(pop)
        imdb = getattr(row, "imdb_id", None)
        if imdb is not None and not pd.isna(imdb):
            # a number, or digits as text (new films); anything else (a person's "nm..." id in the source
            # data) is not an IMDb title number and is skipped, never a reason to fail an import
            with contextlib.suppress(ValueError):
                by_imdb[int(imdb)] = mid
        variants = {norm(title), norm(_TRAILING_TAG.sub("", title))}
        for alt in re.findall(r"\(([^()]{4,})\)", title):
            variants.add(norm(re.sub(r"^a\.k\.a\.?\s*", "", alt, flags=re.IGNORECASE)))
        for v in variants - {""}:
            if mid not in by_title[v]:
                by_title[v].append(mid)
            for w in v.split():
                if w not in _STOP and not w.isdigit():
                    by_word[w].add(v)
    return Catalogue(years, pops, by_imdb, dict(by_title), dict(by_word))


def _pick(cat: Catalogue, ids: list[int], year: int | None, at: datetime | None) -> int | None:
    """Several films share a title (remakes): the one from the given year (or a year off), else the most
    watched film released before it was watched."""
    if not ids:
        return None
    if year is not None:
        close = [m for m in ids if cat.year[m] is not None and abs(cat.year[m] - year) <= 1]
        if not close:
            return None
        return min(close, key=lambda m: (abs(cat.year[m] - year), -cat.popularity[m]))
    if at is not None:
        released = [m for m in ids if cat.year[m] is None or cat.year[m] <= at.year]
        ids = released or ids
    return max(ids, key=lambda m: cat.popularity[m])


def _fuzzy(cat: Catalogue, key: str) -> list[int]:
    """A near-identical title ("Spider-Man Into the Spider Verse" for "...Spider-Verse"), or one that
    only adds a couple of words ("Star Wars: The Force Awakens" for "Star Wars: Episode VII - The Force
    Awakens")."""
    words = [w for w in key.split() if w not in _STOP and not w.isdigit()]
    if len(key.split()) < 2 or not words:
        return []
    candidates = set.intersection(*(cat.by_word.get(w, set()) for w in words))
    query = set(key.split())
    best, best_score = None, 0.0
    for c in candidates:
        extra = set(c.split()) - query
        ratio = SequenceMatcher(None, key, c).ratio()
        score = ratio if ratio >= 0.9 else (0.85 if len(extra) <= 2 and query <= set(c.split()) else 0.0)
        if score > best_score:
            best, best_score = c, score
    return cat.by_title[best] if best else []


def match_row(cat: Catalogue, row: Row) -> int | None:
    if row.imdb is not None and row.imdb in cat.by_imdb:
        return cat.by_imdb[row.imdb]
    year = row.year
    if year is None and (y := re.search(r"\((\d{4})\)\s*$", row.title)):  # "Dune (2021)"
        year = int(y.group(1))
    for key in dict.fromkeys([norm(row.title), norm(_TRAILING_TAG.sub("", row.title))]):
        if key and key in cat.by_title:
            found = _pick(cat, cat.by_title[key], year, row.at)
            if found is not None:
                return found
    key = norm(_TRAILING_TAG.sub("", row.title))
    return _pick(cat, _fuzzy(cat, key), year, row.at) if key else None


def _later(a: datetime | None, b: datetime | None) -> bool:
    """Is a later than b? A known date beats an unknown one."""
    return a is not None and (b is None or a > b)


def preview(name: str, content: str, encoding: Literal["text", "base64"], items: pd.DataFrame) -> Preview:
    """Read an export and match it to the catalogue. Nothing is saved.

    One film can appear several times (rewatches; rated and also on the watchlist). Then a rating beats a
    viewing, which beats My list; the date kept is the latest viewing or rating (the latest addition for
    a My list film), and the rating kept is the most recent one."""
    source, rows, counts = parse(read_upload(name, content, encoding))
    cat = catalogue(items)
    merged: dict[int, Match] = {}
    rated_at: dict[int, datetime | None] = {}
    unmatched: dict[str, None] = {}
    tv = counts["tv"]
    for row in rows:
        if not row.title and row.imdb is None:
            continue
        movie_id = match_row(cat, row)
        if movie_id is None:
            if source == "netflix" and looks_like_episode(row.title):
                tv += 1
            else:
                unmatched.setdefault(row.title)
            continue
        m = merged.get(movie_id)
        if m is None:
            m = merged[movie_id] = Match(movie_id, row.kind, row.title, times=0)
        if KIND_ORDER[row.kind] < KIND_ORDER[m.kind]:
            if m.kind == "watchlist":
                m.at = None  # a My list date no longer applies once the film was watched
            m.kind, m.source_title = row.kind, row.title
        if row.kind != "watchlist":
            m.times += 1
        if (row.kind != "watchlist" or m.kind == "watchlist") and _later(row.at, m.at):
            m.at = row.at
        if row.rating is not None and (movie_id not in rated_at or not _later(rated_at[movie_id], row.at)):
            m.rating, rated_at[movie_id] = row.rating, row.at
    for m in merged.values():
        m.times = max(1, m.times)
    matches = sorted(merged.values(), key=lambda m: (m.at is None, -(m.at.timestamp() if m.at else 0)))
    return Preview(source, len(rows), matches, tv, counts["short"], list(unmatched))
