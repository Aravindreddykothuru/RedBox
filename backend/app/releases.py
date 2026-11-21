"""New releases, kept up to date while the API runs.

The models learned from MovieLens ratings, which end in October 2023. Newer films reach the app in two
steps, both without retraining:

1. **Arrival** (once a day, `Releases.refresh`): Wikidata is asked for films released this year and last in
   the six languages (`recsys.data.wikipedia.RELEASES`); the ones the app doesn't have yet are read from
   Wikipedia (story and poster), checked (`usable_films`), embedded, and placed in the iALS taste space from
   their story (`recsys.serving.newfilms.add_releases`). They are recommended from that moment, and saved
   into the bundle so a restart keeps them.
2. **Warm-up** (after every rating of a new film, `Releases.update_feedback`): each new film's factors are
   re-estimated from its story's guess and the tastes of the app users who rated it (collaborative topic
   regression, `Engine.set_feedback`), so it moves towards the viewers who liked it.

Once enough app users like a new film, retraining with the app's ratings (`recsys import-app-ratings`,
then `ml/scripts/train_app_models.sh --refresh`) makes it a regular film in every model.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

from recsys.data.wikipedia import NEW_ID_OFFSET, RELEASES, wikidata_films, with_pages
from recsys.serving.engine import Engine
from sqlalchemy import select

from app.db import Database, Rating, WatchedItem
from app.services import user_history
from app.settings import Settings

log = logging.getLogger("redbox.releases")
RECHECK_AFTER = timedelta(days=7)  # a film turned away (no article text yet, say) is looked at again then


class Releases:
    def __init__(self, engine: Engine, db: Database, settings: Settings):
        self.engine, self.db, self.settings = engine, db, settings
        self.refreshed_at: datetime | None = None
        self.last_added = 0
        self.added_total = 0
        self.last_error: str | None = None
        self.rated_films = 0
        self._turned_away: dict[int, datetime] = {}
        self._refresh_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------------------------------ warm-up
    def update_feedback(self) -> int:
        """Recompute every new film's factors from the app's ratings of new films (all of them, so the
        result never depends on the order of updates). Returns how many new films have ratings."""
        engine = self.engine
        with self._feedback_lock, self.db.sessions() as s:
            opinions: dict[tuple, bool] = {}
            for user, film in s.execute(
                select(WatchedItem.user_id, WatchedItem.movie_id).where(WatchedItem.movie_id >= NEW_ID_OFFSET)
            ):
                opinions[(user, film)] = True  # watched: liked, unless a rating says otherwise
            for user, film, stars in s.execute(
                select(Rating.user_id, Rating.movie_id, Rating.rating).where(Rating.movie_id >= NEW_ID_OFFSET)
            ):
                opinions[(user, film)] = stars >= self.settings.positive_threshold
            histories = {
                user: user_history(s, engine, user, self.settings.positive_threshold)[0]
                for user in {u for u, _ in opinions}
            }
        feedback: dict[int, list[tuple[list[int], bool]]] = defaultdict(list)
        for (user, film), liked in opinions.items():
            idx = engine.movie_to_idx.get(film)
            if idx is not None and idx >= engine.n_items:
                feedback[idx].append((histories[user], liked))
        engine.set_feedback(feedback)
        self.rated_films = len(feedback)
        return self.rated_films

    # ------------------------------------------------------------------------------------------ arrival
    def fetch(self, today: date | None = None):
        """This year's and last year's releases in every language that the app doesn't have yet, with
        their Wikipedia pages. Films turned away recently are skipped until RECHECK_AFTER."""
        import pandas as pd

        today = today or date.today()
        present = set(self.engine.items["movie_id"].astype("int64"))
        now = datetime.now(UTC)
        tables = []
        for language, (first, min_links) in RELEASES.items():
            table = wikidata_films(language, max(first, today.year - 1), today.year, min_links)
            if table.empty:
                continue
            ids = [NEW_ID_OFFSET + int(q[1:]) for q in table["wikidata"]]
            fresh = [
                m not in present and now - self._turned_away.get(m, now - 2 * RECHECK_AFTER) > RECHECK_AFTER
                for m in ids
            ]
            tables.append(table.loc[fresh])  # a list of bools: never empty here (table isn't)
        tables = [t for t in tables if not t.empty]
        if not tables:
            return pd.DataFrame()
        return with_pages(pd.concat(tables, ignore_index=True).drop_duplicates("wikidata"))

    def refresh(self, encoder=None) -> int:
        """Add new releases (see the module's docstring). Returns how many were added."""
        from recsys.serving.newfilms import add_releases, item_encoder, write_new_items

        with self._refresh_lock:
            try:
                films = self.fetch()
                added = 0
                if not films.empty:
                    encoder = encoder or item_encoder(self.engine)
                    rows = add_releases(self.engine, films, encoder)
                    added = len(rows)
                    taken = set(rows["movie_id"].tolist())
                    stamp = datetime.now(UTC)
                    for q in films["wikidata"]:
                        if NEW_ID_OFFSET + int(q[1:]) not in taken:
                            self._turned_away[NEW_ID_OFFSET + int(q[1:])] = stamp
                    if added and self.settings.releases_save:
                        write_new_items(self.engine)
                self.last_added, self.added_total = added, self.added_total + added
                self.refreshed_at, self.last_error = datetime.now(UTC), None
                log.info("new releases: %d added (%d checked)", added, len(films))
                return added
            except Exception as e:  # the next day's run tries again; the app keeps serving meanwhile
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("new releases: refresh failed")
                return 0

    # ------------------------------------------------------------------------------------------ schedule
    def start(self) -> None:
        """Refresh in a background thread: first after a short delay (the API answers at once), then every
        REDBOX_RELEASES_REFRESH_HOURS. 0 switches it off (tests, or a machine without internet)."""
        hours = self.settings.releases_refresh_hours
        if hours <= 0 or self._thread is not None:
            return

        def loop() -> None:
            wait = self.settings.releases_first_delay_seconds
            while not self._stop.wait(wait):
                self.refresh()
                wait = hours * 3600

        self._thread = threading.Thread(target=loop, name="releases", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        engine = self.engine
        return {
            "new_films": len(engine.items) - engine.n_items,
            "languages": engine.new_languages,
            "refreshed_at": self.refreshed_at,
            "last_added": self.last_added,
            "added_since_start": self.added_total,
            "films_with_ratings": self.rated_films,
            "every_hours": self.settings.releases_refresh_hours,
            "last_error": self.last_error,
        }
