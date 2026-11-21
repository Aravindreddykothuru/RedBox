"""Posters and summaries from Wikipedia (no API key needed), for a dataset's films and for new films.

Wikidata links a film's IMDb id (property P345) to its English Wikipedia article. The Wikipedia API then gives
the article's lead image (the theatrical poster on almost every film article) and its opening section as
plain text. `fetch_wiki` saves data/processed/<dataset>/wiki.parquet, which is used:

- by `recsys content`: the summary becomes part of each film's text, for semantic search and "similar story";
- by `recsys content --bundle` and `recsys export`: the poster URL and the summary go into the bundle's items.

`language_films` lists recent films in one language (English, Hindi, Telugu, Tamil, Kannada or Malayalam)
the same way, with their genres, for the new-film cold start (`recsys newfilms`, and the API's daily
refresh of new releases).

Wikipedia's text is CC BY-SA, and the app credits it. The posters are the studios' artwork, which Wikipedia
shows under a fair-use rationale. That is fine for this research demo, but a public deployment should
prefer TMDB artwork (`recsys tmdb`, with a free key).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

from recsys.paths import processed_dir

log = logging.getLogger(__name__)

USER_AGENT = "RedBoxRecommender/1.0 (educational film recommender; python-requests)"
WIKIDATA = "https://query.wikidata.org/sparql"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
SUMMARY_CHARS = 1500


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get(session: requests.Session, url: str, params: dict, tries: int = 5) -> dict:
    """GET JSON, waiting and retrying when the service says it is busy (429, 5xx)."""
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=120)
        except requests.ConnectionError:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After") or 0) or 5 * 2**attempt
            log.warning("%s answered %d; waiting %.0fs", url, r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"{url} kept failing after {tries} tries")


def _article(url: str) -> str:
    return unquote(url.rsplit("/wiki/", 1)[1]).replace("_", " ")


def wikidata_articles(imdb_ids: list[str], batch: int = 200) -> dict[str, tuple[str, str]]:
    """IMDb id ("tt0114709") -> (Wikidata id, English Wikipedia article title)."""
    s = _session()
    out: dict[str, tuple[str, str]] = {}
    for i in range(0, len(imdb_ids), batch):
        values = " ".join(f'"{x}"' for x in imdb_ids[i : i + batch])
        query = f"""SELECT ?imdb ?film ?article WHERE {{
          VALUES ?imdb {{ {values} }}
          ?film wdt:P345 ?imdb .
          ?article schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> .
        }}"""
        for b in _get(s, WIKIDATA, {"query": query, "format": "json"})["results"]["bindings"]:
            out.setdefault(
                b["imdb"]["value"], (b["film"]["value"].rsplit("/", 1)[1], _article(b["article"]["value"]))
            )
        log.info("Wikidata: %d / %d IMDb ids looked up", min(i + batch, len(imdb_ids)), len(imdb_ids))
    return out


def _clean_summary(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(
        r"\s*\([^()]*(?:pronunciation|listen|lit\.|transl\.)[^()]*\)", "", text
    )  # pronunciation notes
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > SUMMARY_CHARS:  # end on a full sentence
        cut = text[:SUMMARY_CHARS]
        text = cut[: cut.rfind(". ") + 1] or cut
    return text or None


def wikipedia_pages(
    titles: list[str],
    batch: int = 20,
    pause: float = 1.5,
    cache: Path | None = None,
) -> dict[str, dict]:
    """Article title -> {"poster": image URL or None, "summary": opening section or None}. Redirects and
    title normalisation are followed back to the title asked for. A steady ``pause`` between requests
    keeps within Wikipedia's rate limit (bursts get HTTP 429). With ``cache`` (a JSON file), finished
    articles are saved as it goes and skipped on a rerun."""
    s = _session()
    out: dict[str, dict] = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    titles = [t for t in titles if t not in out]
    for i in range(0, len(titles), batch):
        if i:
            time.sleep(pause)
        chunk = titles[i : i + batch]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "pageimages|extracts",
            "piprop": "thumbnail",
            "pithumbsize": 342,
            "pilicense": "any",  # film posters are non-free files: without this none is returned
            "pilimit": batch,
            "exintro": 1,
            "explaintext": 1,
            "exlimit": batch,
            "redirects": 1,
            "titles": "|".join(chunk),
        }
        found: dict[str, dict] = {}
        while True:
            data = _get(s, WIKIPEDIA, params)
            query = data.get("query", {})
            alias = {a["from"]: a["to"] for a in query.get("normalized", []) + query.get("redirects", [])}
            for page in query.get("pages", []):
                entry = found.setdefault(page["title"], {"poster": None, "summary": None})
                thumb = page.get("thumbnail", {}).get("source")
                if thumb:
                    entry["poster"] = thumb.split("?", 1)[0]
                if page.get("extract"):
                    entry["summary"] = _clean_summary(page["extract"])
            if "continue" not in data:
                break
            params = {**params, **data["continue"]}
        for t in chunk:
            final = t
            for _ in range(3):  # normalised, then redirected
                final = alias.get(final, final)
            out[t] = found.get(final, {"poster": None, "summary": None})
        log.info("Wikipedia: %d / %d articles read", min(i + batch, len(titles)), len(titles))
        if cache and (i // batch) % 10 == 9:
            cache.write_text(json.dumps(out), encoding="utf-8")
    if cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def fetch_wiki(dataset: str, refresh: bool = False) -> pd.DataFrame:
    """Poster and summary for every film of a dataset, saved as data/processed/<dataset>/wiki.parquet.
    Films already fetched are kept unless ``refresh``."""
    out_dir = processed_dir(dataset)
    target = out_dir / "wiki.parquet"
    links = pd.read_parquet(out_dir / "links.parquet")
    links = links[links["imdb_id"].notna()].copy()
    links["imdb"] = "tt" + links["imdb_id"].astype(str).str.zfill(7)
    done = pd.read_parquet(target) if target.exists() and not refresh else None
    todo = links if done is None else links[~links["item_id"].isin(done["item_id"])]
    if len(todo):
        articles = wikidata_articles(todo["imdb"].tolist())
        titles = sorted({t for _, t in articles.values()})
        pages = wikipedia_pages(titles, cache=out_dir / "wiki_pages.json")  # resumable
        rows = []
        for item, imdb in zip(todo["item_id"], todo["imdb"], strict=True):
            qid, title = articles.get(imdb, (None, None))
            page = pages.get(title, {}) if title else {}
            rows.append(
                {
                    "item_id": int(item),
                    "wikidata": qid,
                    "wiki_title": title,
                    "poster_url": page.get("poster"),
                    "summary": page.get("summary"),
                }
            )
        new = pd.DataFrame(rows)
        done = new if done is None else pd.concat([done, new], ignore_index=True)
    done.to_parquet(target, index=False)
    return done


GENRE_MAP = {  # Wikidata genre labels -> the catalogue's genres
    "action film": "Action",
    "action thriller": "Thriller",
    "adventure film": "Adventure",
    "animated film": "Animation",
    "biographical film": "Drama",
    "children's film": "Children's",
    "comedy film": "Comedy",
    "comedy-drama": "Comedy",
    "crime film": "Crime",
    "crime thriller film": "Thriller",
    "documentary film": "Documentary",
    "drama film": "Drama",
    "epic film": "Adventure",
    "family film": "Children's",
    "fantasy film": "Fantasy",
    "historical film": "Drama",
    "horror film": "Horror",
    "masala film": "Action",
    "musical film": "Musical",
    "mystery film": "Mystery",
    "neo-noir": "Film-Noir",
    "period drama": "Drama",
    "political film": "Drama",
    "romance film": "Romance",
    "romantic comedy": "Romance",
    "science fiction film": "Sci-Fi",
    "sports film": "Drama",
    "spy film": "Thriller",
    "superhero film": "Action",
    "thriller film": "Thriller",
    "war film": "War",
    "western film": "Western",
}

LANGUAGES = {
    "English": "Q1860",
    "Hindi": "Q1568",
    "Telugu": "Q8097",
    "Tamil": "Q5885",
    "Kannada": "Q33673",
    "Malayalam": "Q36236",
}

# Which new films each language brings, as (first release year, fewest Wikipedia language editions).
# MovieLens's ratings end in October 2023, so English and Hindi films are new from 2023; MovieLens barely
# covers South Indian cinema, so those are new from 2016. English has thousands of films a year: an
# article in at least 8 language editions keeps the ones people talk about (about 1,000 of 2,600).
RELEASES = {
    "English": (2023, 8),
    "Hindi": (2023, 2),
    "Telugu": (2016, 0),
    "Tamil": (2016, 0),
    "Kannada": (2016, 0),
    "Malayalam": (2016, 0),
}


def wikidata_films(language: str, since: int, until: int, min_links: int = 0) -> pd.DataFrame:
    """Feature films whose original language is ``language`` (a key of LANGUAGES), first released from
    ``since`` to ``until`` and no later than today, that have an English Wikipedia article and at least
    ``min_links`` Wikipedia language editions: Wikidata id, IMDb id, article title, year, genres (mapped to
    the catalogue's) and ``links`` (how many Wikipedias cover it: a measure of how widely it is known).
    One query to Wikidata; no Wikipedia pages yet (see with_pages)."""
    qid = LANGUAGES[language]
    query = f"""SELECT ?film ?imdb ?article ?year ?links
      (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
    WHERE {{
      ?film wdt:P31 wd:Q11424 ; wdt:P364 wd:{qid} ; wdt:P577 ?date ; wikibase:sitelinks ?links .
      ?article schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> .
      OPTIONAL {{ ?film wdt:P345 ?imdb . }}
      OPTIONAL {{ ?film wdt:P136 ?genre . ?genre rdfs:label ?genreLabel . FILTER(LANG(?genreLabel) = "en") }}
      BIND(YEAR(?date) AS ?year)
      FILTER(?year >= {since} && ?year <= {until} && ?date <= NOW() && ?links >= {min_links})
    }} GROUP BY ?film ?imdb ?article ?year ?links"""
    s = _session()
    rows: dict[str, dict] = {}
    for b in _get(s, WIKIDATA, {"query": query, "format": "json"})["results"]["bindings"]:
        film = b["film"]["value"].rsplit("/", 1)[1]
        year = int(b["year"]["value"])
        genres = {
            GENRE_MAP[g.lower()]
            for g in b.get("genres", {}).get("value", "").split("|")
            if g.lower() in GENRE_MAP
        }
        row = rows.setdefault(
            film,
            {
                "wikidata": film,
                "imdb": b.get("imdb", {}).get("value"),
                "wiki_title": _article(b["article"]["value"]),
                "year": year,
                "genres": set(),
                "language": language,
                "links": int(b["links"]["value"]),
            },
        )
        row["year"] = min(row["year"], year)  # several release dates: the first
        row["genres"] |= genres
    table = pd.DataFrame(rows.values())
    if not table.empty:
        table["genres"] = table["genres"].map(sorted)
    return table


def with_pages(table: pd.DataFrame, cache: Path | None = None) -> pd.DataFrame:
    """``table`` (from wikidata_films) with each article's poster, summary and a display title."""
    table = table.copy()
    if table.empty:
        return table.assign(poster_url=None, summary=None, title=None)
    pages = wikipedia_pages(table["wiki_title"].tolist(), cache=cache)
    table["poster_url"] = [pages.get(t, {}).get("poster") for t in table["wiki_title"]]
    table["summary"] = [pages.get(t, {}).get("summary") for t in table["wiki_title"]]
    table["title"] = table["wiki_title"].str.replace(r"\s*\((?:[^()]*\bfilm|\d{4}[^()]*)\)$", "", regex=True)
    return table


def language_films(
    language: str, since: int, until: int, cache: Path | None = None, min_links: int = 0
) -> pd.DataFrame:
    """wikidata_films with their Wikipedia pages. ``cache``: a JSON file of fetched articles, so an
    interrupted run resumes where it stopped (see wikipedia_pages)."""
    return with_pages(wikidata_films(language, since, until, min_links), cache=cache)


# "X is a 2019 Indian Tamil-language action film" (or "...family drama", "...documentary"), not "X is an
# Indian film director", "a film production studio" or "known for comedy films"
_FILM_LEAD = re.compile(
    r"\b(?:is|was)\s+(?:(?:an?|the)\s+)?[^.]{0,160}?"
    r"\b(?:film|movie|documentary|drama|comedy|thriller|horror|romance|satire|fantasy|musical|western)\b"
    r"(?!\s*(?:director|producer|production|studio|distribut|editor|actor|actress|composer|industry|critic|"
    r"maker|company|festival|award|personality|journalist|lyricist|writer|films|movies))",
    re.IGNORECASE,
)
# a person's article: "(born 1985" or life dates "(23 July 1949 - ...)", with an en dash; or a series
_NOT_A_FILM = re.compile(
    r"\((?:born\b|\d{1,2}\s+\w+\s+\d{4}\s*[\N{EN DASH}-])|\b(?:television|web|tv)\s+series\b", re.IGNORECASE
)
_LEAD_YEAR = re.compile(r"\b(?:is|was)\s+(?:an?\s+)?((?:19|20)\d\d)\b")


def describes_film(summary: str | None, year: int | None = None) -> bool:
    """Whether a Wikipedia summary is about a film (released about ``year`` or later). A film's Wikidata
    item can link to an article about something else: a shelved film's article becomes a redirect to its
    lead actor, its studio or a list of films, and a sequel's to the original. Its text would describe the
    wrong thing, so such films are left out. The opening sentence must call the subject a film (or a
    drama, a documentary...), not a person or a series; and the film it describes may be released later
    (delays are common) but not more than two years earlier (that is another, older film)."""
    if not isinstance(summary, str) or not summary.strip():
        return False
    lead = summary.strip()[:300]  # the opening (no sentence split: titles have full stops, "Mr.", "No. 150")
    if _NOT_A_FILM.search(lead) or not _FILM_LEAD.search(lead):
        return False
    said = _LEAD_YEAR.search(lead)
    return year is None or said is None or int(said.group(1)) >= int(year) - 2


def save(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


NEW_ID_OFFSET = 1_000_000_000  # new films' ids: this + the Wikidata number (Q12345 -> 1000012345)


def new_film_catalogue(folder: Path | None = None) -> pd.DataFrame:
    """Every new film fetched so far (`recsys newfilms`: data/processed/newfilms/<language>_<years>.parquet),
    once each, by its app id: movie_id, title, year, genres, imdb, summary, poster_url, language. Used when
    app ratings make a new film part of the training data (it "graduates" into the catalogue)."""
    from recsys.paths import data_dir

    folder = Path(folder or data_dir() / "processed" / "newfilms")
    tables = [pd.read_parquet(f) for f in sorted(folder.glob("*_*-*.parquet"))] if folder.exists() else []
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame(columns=["movie_id", "title", "year", "genres", "imdb", "summary", "poster_url"])
    films = pd.concat(tables, ignore_index=True)
    films["movie_id"] = [NEW_ID_OFFSET + int(q[1:]) for q in films["wikidata"]]
    return films.drop_duplicates("movie_id").reset_index(drop=True)
