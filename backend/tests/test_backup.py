"""The SQLite backup for hosts without a permanent disk: restored at startup, uploaded when it changed."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.backup import Backup
from app.settings import Settings
from tests.conftest import make_client


class FakeHub:
    """A dataset repository kept in a local folder, standing in for huggingface_hub."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.uploads = 0

    def install(self, monkeypatch) -> None:
        import huggingface_hub

        hub = self

        class Api:
            def __init__(self, token=None):
                pass

            def create_repo(self, *args, **kwargs):
                pass

            def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
                hub.uploads += 1
                shutil.copyfile(path_or_fileobj, hub.folder / path_in_repo)

        def download(repo_id, filename, repo_type=None, token=None):
            from huggingface_hub.errors import EntryNotFoundError

            path = hub.folder / filename
            if not path.exists():
                raise EntryNotFoundError("no file")
            return str(path)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    folder = tmp_path / "hub"
    folder.mkdir()
    fake = FakeHub(folder)
    fake.install(monkeypatch)
    return fake


def settings_for(db: Path, **extra) -> Settings:
    return Settings(database_url=f"sqlite:///{db.as_posix()}", backup_repo="someone/redbox-data", **extra)


def test_off_without_a_repo_a_token_or_a_sqlite_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not Backup(Settings(database_url="sqlite:///:memory:", backup_repo="x/y"), token="t").enabled
    assert not Backup(settings_for(tmp_path / "a.db"), token=None).enabled
    assert not Backup(Settings(database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}"), token="t").enabled
    assert not Backup(
        Settings(database_url="postgresql+psycopg://u:p@h/db", backup_repo="x/y"), token="t"
    ).enabled


def test_upload_only_when_changed_and_restore_into_a_fresh_disk(tmp_path, hub):
    db = tmp_path / "live" / "redbox.db"
    db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ratings (user TEXT, film INTEGER, stars REAL)")
        conn.execute("INSERT INTO ratings VALUES ('a', 1, 5.0)")
    b = Backup(settings_for(db), token="t")
    assert b.enabled and not b.restore()  # nothing to restore yet
    assert b.save() and hub.uploads == 1
    assert not b.save() and hub.uploads == 1  # unchanged: no upload
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO ratings VALUES ('b', 2, 4.0)")
    assert b.save() and hub.uploads == 2

    fresh = tmp_path / "after-restart" / "redbox.db"  # the Space restarted: an empty disk
    again = Backup(settings_for(fresh), token="t")
    assert again.restore()
    with sqlite3.connect(fresh) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 2
    assert not again.save()  # what it restored is what the repository has


def test_accounts_survive_a_restart(bundle, tmp_path, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    first = tmp_path / "one" / "redbox.db"
    first.parent.mkdir()
    options = {"auth_mode": "local", "backup_repo": "someone/redbox-data"}
    with make_client(bundle, database_url=f"sqlite:///{first.as_posix()}", **options) as c:
        r = c.post("/auth/register", json={"email": "keep@example.com", "password": "Long-enough-pass-1"})
        assert r.status_code == 201
    # shutting down uploaded the database; a new disk gets it back at startup
    second = tmp_path / "two" / "redbox.db"
    with make_client(bundle, database_url=f"sqlite:///{second.as_posix()}", **options) as c:
        r = c.post("/auth/login", json={"email": "keep@example.com", "password": "Long-enough-pass-1"})
        assert r.status_code == 200
