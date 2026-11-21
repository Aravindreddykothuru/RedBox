"""Keep the API's SQLite database safe on a host without a permanent disk.

A free Hugging Face Space loses its files whenever it restarts (after a rebuild, or after sleeping through
48 hours without visitors), and with them every account and rating. With REDBOX_BACKUP_REPO set (a
*private* Hugging Face dataset, e.g. "you/redbox-data") and a write token in HF_TOKEN (a Space secret):

- at startup, before the database opens, the latest copy is downloaded from the dataset;
- every REDBOX_BACKUP_MINUTES, if anything changed, a consistent snapshot (SQLite's online backup, safe
  while requests write) is uploaded; and once more when the API shuts down.

At most one upload per interval, and none when nothing changed. A hosted Postgres (REDBOX_DATABASE_URL,
see docs/DEPLOYMENT.md) needs none of this: the backup only runs for a SQLite database.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

from sqlalchemy.engine import make_url

from app.settings import Settings

log = logging.getLogger("redbox.backup")
FILE = "redbox.db"


class Backup:
    def __init__(self, settings: Settings, token: str | None = None):
        url = make_url(settings.database_url)
        self.path = Path(url.database) if url.get_backend_name() == "sqlite" and url.database else None
        self.repo = settings.backup_repo
        self.minutes = settings.backup_minutes
        self.token = token or os.environ.get("HF_TOKEN")
        self.enabled = bool(self.repo and self.token and self.path and str(self.path) != ":memory:")
        self.last_digest: str | None = None
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def restore(self) -> bool:
        """Download the latest copy (if there is one) before the database opens. True if restored."""
        if not self.enabled:
            return False
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        api = self._api()
        try:
            api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
            copy = hf_hub_download(self.repo, FILE, repo_type="dataset", token=self.token)
        except (EntryNotFoundError, RepositoryNotFoundError):
            log.info("backup: no copy in %s yet; starting with an empty database", self.repo)
            return False
        except Exception as e:  # never block startup: the API runs, and the next save writes a copy
            self.last_error = f"{type(e).__name__}: {e}"
            log.exception("backup: restore failed")
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(copy, self.path)
        self.last_digest = _digest(self.path)
        log.info("backup: restored %s from %s", self.path, self.repo)
        return True

    def save(self) -> bool:
        """Upload a snapshot if the database changed since the last one. True if uploaded."""
        if not self.enabled or not self.path.exists():
            return False
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / FILE
            source = sqlite3.connect(self.path)
            target = sqlite3.connect(snapshot)
            try:
                source.backup(target)  # consistent even while requests are writing
            finally:
                target.close()
                source.close()
            digest = _digest(snapshot)
            if digest == self.last_digest:
                return False
            try:
                self._api().upload_file(
                    path_or_fileobj=str(snapshot),
                    path_in_repo=FILE,
                    repo_id=self.repo,
                    repo_type="dataset",
                    commit_message="Database backup",
                )
            except Exception as e:  # tried again at the next interval
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("backup: upload failed")
                return False
            self.last_digest, self.last_error = digest, None
            log.info("backup: uploaded to %s", self.repo)
            return True

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(self.minutes * 60):
                self.save()

        self._thread = threading.Thread(target=loop, name="backup", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.save()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
