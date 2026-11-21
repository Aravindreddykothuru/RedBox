"""Film posters, served by the API from a disk cache.

The posters come from Wikipedia's image server (upload.wikimedia.org). It limits automated clients to
roughly one image every one to three seconds per address (HTTP 429 beyond that) while browsers get far
more, and it can be slow or throttled at any time. So:
- a cached poster (342 px wide WebP, about 20 KB) is served by the API with a year-long cache header;
- a poster not cached yet: the browser is sent to Wikipedia's file for this view (its own, generous,
  browser limits), and the poster joins a background queue that caches it, one at a time at the pace
  Wikimedia accepts;
- `python -m app.posters` fills the whole cache ahead of time (stoppable and resumable).
The website draws each film's typographic cover under its poster, so a poster that never arrives leaves
the cover, never an empty box.
"""

from __future__ import annotations

import argparse
import io
import itertools
import logging
import os
import queue
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

log = logging.getLogger(__name__)

WIDTH = 342  # the cards' poster width; a film page shows it at most ~260 px wide
QUALITY = 82
USER_AGENT = "RedBoxRecommender/1.0 (film recommender; poster cache; python-requests)"
# Pacing. Wikimedia's image server allows somewhere around one request every one to three seconds from
# one address, and answers faster requests with HTTP 429 ("Retry-After: 1"). Its limit isn't published and
# changes, so the pace adapts: each 429 makes the gap between requests 1.5 times longer (up to 10 s), each
# success 5% shorter (down to MIN_INTERVAL).
MIN_INTERVAL = 1.0
MAX_INTERVAL = 10.0
MAX_WAIT = 120.0  # longest one poster waits through 429s before it is skipped (nobody waits for it)
QUEUE_LIMIT = 20_000


class PosterUnavailable(Exception):
    """The source would not give the poster now (throttled, gone or not an image): try again later."""


_session: requests.Session | None = None
_upstream = threading.Lock()  # one request to Wikimedia at a time, across the whole process
_last = 0.0
_pace = MIN_INTERVAL  # the current gap between requests (adaptive, see above)
_queue: queue.Queue[tuple[Path, str]] = queue.Queue()  # posters seen uncached, for the background cacher
_queued: set[Path] = set()
_queued_lock = threading.Lock()
_worker: threading.Thread | None = None


def _get(url: str, max_wait: float = MAX_WAIT) -> bytes:
    """The bytes at ``url``, fetched politely: spaced from the previous request by the adaptive pace, and
    on 429 or a server error retried after as long as Retry-After says (at most ``max_wait`` s in all)."""
    global _session, _last, _pace
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    waited = 0.0
    with _upstream:
        for attempt in range(50):
            gap = _pace - (time.monotonic() - _last)  # spaced from the previous request's start
            if gap > 0:
                time.sleep(gap)
            _last = time.monotonic()
            try:
                r = _session.get(url, timeout=20)
            except requests.RequestException as e:
                raise PosterUnavailable(str(e)) from e
            if r.status_code == 200:
                _pace = max(MIN_INTERVAL, _pace * 0.95)
                return r.content
            if r.status_code not in (429, 500, 502, 503, 504):
                raise PosterUnavailable(f"HTTP {r.status_code}")
            if r.status_code == 429:
                _pace = min(MAX_INTERVAL, _pace * 1.5)
            try:
                wait = float(r.headers.get("retry-after", ""))  # what the server asks for
            except ValueError:
                wait = min(2.0**attempt, MAX_INTERVAL)  # 1, 2, 4, 8 s without a header
            if waited + wait > max_wait:
                break
            time.sleep(wait)
            waited += wait
    raise PosterUnavailable("throttled")


def to_webp(data: bytes, width: int = WIDTH) -> bytes:
    """A card-sized WebP of any image: at most ``width`` wide, proportions kept, transparency on black."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # Pillow raises several kinds for a file that is not an image
        raise PosterUnavailable(f"not an image: {e}") from e
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        backing = Image.new("RGBA", img.size, (11, 9, 8, 255))
        img = Image.alpha_composite(backing, img)
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, "WEBP", quality=QUALITY, method=4)
    return out.getvalue()


def save(path: Path, url: str, max_wait: float = MAX_WAIT) -> None:
    """Fetch ``url``, convert it and write it to ``path`` (atomically: a reader never sees half a file)."""
    webp = to_webp(_get(url, max_wait))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
    tmp.write_bytes(webp)
    tmp.replace(path)


def enqueue(path: Path, url: str) -> None:
    """Cache this poster in the background (once, however often it is asked for meanwhile)."""
    global _worker
    with _queued_lock:
        if path in _queued or len(_queued) >= QUEUE_LIMIT:
            return
        _queued.add(path)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_cache_forever, name="poster-cacher", daemon=True)
            _worker.start()
    _queue.put((path, url))


def _cache_forever() -> None:
    while True:
        path, url = _queue.get()
        try:
            if not path.exists():
                save(path, url)
        except PosterUnavailable as e:
            log.info("poster %s not cached: %s", path.name, e)
        except Exception:  # a background thread must outlive any one bad file
            log.exception("poster %s not cached", path.name)
        finally:
            with _queued_lock:
                _queued.discard(path)
            _queue.task_done()


def catalogue(bundle: Path) -> pd.DataFrame:
    """movie_id, poster URL and popularity for every film with a Wikipedia poster, the models' films and
    the new films, read from the bundle's tables (without loading any model)."""
    tables = []
    for name in ("items.parquet", "new_items.parquet"):
        path = bundle / name
        if path.exists():
            cols = [c for c in ("movie_id", "poster", "popularity") if c in pd.read_parquet(path).columns]
            tables.append(pd.read_parquet(path, columns=cols))
    films = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["movie_id", "poster"])
    if "poster" not in films:
        return films.iloc[0:0]
    return films.dropna(subset=["poster"])


def warm_order(films: pd.DataFrame) -> list[tuple[int, str]]:
    """(movie_id, poster URL) in the order to fetch: the most-watched films (they fill most rows) taking
    turns with the new films (their own row on everyone's home page), so both are covered early."""
    pop = pd.to_numeric(films["popularity"], errors="coerce") if "popularity" in films else 0
    films = films.assign(popularity=pop).fillna({"popularity": 0})
    is_new = films["movie_id"] >= 1_000_000_000  # new films' ids (recsys.serving.newfilms.NEW_ID_OFFSET)
    known = films[~is_new].sort_values("popularity", ascending=False, kind="stable")
    new = films[is_new]
    pairs = itertools.zip_longest(
        zip(known["movie_id"], known["poster"], strict=True), zip(new["movie_id"], new["poster"], strict=True)
    )
    return [(int(m), str(u)) for pair in pairs for item in pair if item is not None for m, u in [item]]


def warm(bundle: Path, cache: Path, limit: int | None = None) -> dict[str, int]:
    """Download every poster the cache lacks, one at a time at the pace Wikimedia accepts (see warm_order).
    Safe to stop and rerun: it starts with what is still missing."""
    order = warm_order(catalogue(bundle))
    todo = [(m, u) for m, u in order if not (cache / f"{m}.webp").exists()]
    if limit is not None:
        todo = todo[:limit]
    counts = {"to fetch": len(todo), "saved": 0, "unavailable": 0}
    log.info("%d posters cached, %d to fetch", len(order) - len(todo), len(todo))
    for n, (movie_id, url) in enumerate(todo, 1):
        try:
            save(cache / f"{movie_id}.webp", url)
            counts["saved"] += 1
        except PosterUnavailable as e:
            counts["unavailable"] += 1
            log.warning("film %d: %s", movie_id, e)
        if n % 100 == 0 or n == len(todo):
            saved, unavailable = counts["saved"], counts["unavailable"]
            log.info("%d / %d: %d saved, %d unavailable", n, len(todo), saved, unavailable)
    return counts


def main() -> None:
    from app.settings import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Fill the poster cache ahead of time.")
    parser.add_argument("--limit", type=int, default=None, help="at most this many posters")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    print(warm(settings.bundle_dir, settings.poster_dir, args.limit))


if __name__ == "__main__":
    main()
