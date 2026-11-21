from __future__ import annotations

import hashlib
import zipfile

import pytest

from recsys.data.download import ChecksumError, _safe_extract, parse_md5_file, verify


def test_parse_md5_both_styles():
    h = "c4d9eecfca2ab87c1945afe126590906"
    assert parse_md5_file(f"MD5 (ml-1m.zip) = {h}\n") == h
    assert parse_md5_file(f"{h.upper()}  ml-1m.zip") == h
    with pytest.raises(ValueError):
        parse_md5_file("not a checksum")


def test_verify_accepts_and_rejects(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello recsys")
    sha = hashlib.sha256(b"hello recsys").hexdigest()
    md5 = hashlib.md5(b"hello recsys").hexdigest()
    assert verify(f, sha256=sha, md5=md5) == sha
    with pytest.raises(ChecksumError, match="SHA-256"):
        verify(f, sha256="0" * 64)
    with pytest.raises(ChecksumError, match="MD5"):
        verify(f, md5="0" * 32)


def test_safe_extract_blocks_zip_slip(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        _safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_normal_archive(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("ml-1m/ratings.dat", "1::1::5::1")
    _safe_extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "ml-1m" / "ratings.dat").read_text() == "1::1::5::1"
