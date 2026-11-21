"""Download, verify and extract dataset archives."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from recsys.data.registry import DatasetSpec
from recsys.paths import raw_dir

_CHUNK = 1 << 20  # 1 MiB


class ChecksumError(RuntimeError):
    pass


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_md5_file(text: str) -> str:
    """Accept both BSD style ``MD5 (file) = <hash>`` and GNU style ``<hash>  file``."""
    match = re.search(r"\b([0-9a-fA-F]{32})\b", text)
    if not match:
        raise ValueError(f"No MD5 hash found in checksum file: {text[:200]!r}")
    return match.group(1).lower()


def verify(path: Path, *, sha256: str | None = None, md5: str | None = None) -> str:
    """Verify ``path`` against whichever checksum is given. Returns the file's SHA-256."""
    if sha256 is None and md5 is None:
        raise ValueError("verify() needs at least one expected checksum")
    actual_sha256 = file_digest(path, "sha256")
    if sha256 is not None and actual_sha256 != sha256.lower():
        raise ChecksumError(f"{path.name}: SHA-256 mismatch (expected {sha256}, got {actual_sha256})")
    if md5 is not None:
        actual_md5 = file_digest(path, "md5")
        if actual_md5 != md5.lower():
            raise ChecksumError(f"{path.name}: MD5 mismatch (expected {md5}, got {actual_md5})")
    return actual_sha256


def _stream_to_file(url: str, dest: Path, timeout: float = 60.0, attempts: int = 8) -> None:
    """Download to ``dest`` via a ``.part`` file. A broken connection resumes where it stopped (HTTP Range
    request), so a large file never has to start over."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:  # nothing left to fetch: the .part file is complete
                    break
                r.raise_for_status()
                resumed = r.status_code == 206
                if not resumed:
                    have = 0  # server ignored the range request: start over
                size = int(r.headers.get("content-length", 0))
                with (
                    tmp.open("ab" if resumed else "wb") as f,
                    tqdm(
                        total=have + size or None, initial=have, unit="B", unit_scale=True, desc=dest.name
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=_CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            if attempt == attempts:
                raise
            print(f"download interrupted ({type(e).__name__}); resuming, attempt {attempt + 1}/{attempts}")
    tmp.replace(dest)  # atomic: never leave a half-written archive under the final name


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing entries that would escape ``target`` (zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(target):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)


def fetch(spec: DatasetSpec, *, force: bool = False) -> Path:
    """Download (if needed), verify and extract a dataset. Returns the extracted folder.
    Zip archives are extracted; single compressed files (e.g. .csv.gz) are kept in the dataset folder."""
    root = raw_dir()
    root.mkdir(parents=True, exist_ok=True)
    is_zip = spec.archive_name.endswith(".zip")
    extracted = root / spec.extracted_dir
    archive = root / spec.archive_name if is_zip else extracted / spec.archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)

    if force or not archive.exists():
        _stream_to_file(spec.url, archive)

    expected_md5 = None
    if spec.sha256 is None and spec.md5_url is not None:
        resp = requests.get(spec.md5_url, timeout=30)
        resp.raise_for_status()
        expected_md5 = parse_md5_file(resp.text)
    try:
        sha256 = verify(archive, sha256=spec.sha256, md5=expected_md5)
    except ChecksumError:
        archive.unlink(missing_ok=True)  # don't keep a bad file around for the next run
        raise
    if spec.sha256 is None:
        print(f"Verified {archive.name} (sha256={sha256}); pin this in registry.py")

    if is_zip and (force or not extracted.exists()):
        _safe_extract(archive, root)
    return extracted
