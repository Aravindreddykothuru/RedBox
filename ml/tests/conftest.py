from __future__ import annotations

from pathlib import Path

import pytest

RATINGS = """\
1::10::5::978300760
1::20::3::978300760
1::30::4::978302109
2::10::2::978301968
2::30::5::978824291
3::20::1::978300275
"""

MOVIES = """\
10::Toy Story (1995)::Animation|Children's|Comedy
20::Misérables, Les (1995)::Drama|Musical
30::Slingshot, The (Kådisbellan ) (1993)::Comedy|Drama
40::Never Rated (2000)::Documentary
"""

USERS = """\
1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
"""


@pytest.fixture
def ml1m_dir(tmp_path: Path) -> Path:
    """A tiny ML-1M lookalike, written in the real Latin-1 encoding."""
    d = tmp_path / "ml-1m"
    d.mkdir()
    for name, text in [("ratings.dat", RATINGS), ("movies.dat", MOVIES), ("users.dat", USERS)]:
        (d / name).write_bytes(text.encode("latin-1"))
    return d
