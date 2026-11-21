"""Watch-history import: reading Netflix, IMDb and Letterboxd exports, matching titles, and what an import
does to recommendations, the profile, data export and account deletion."""

from __future__ import annotations

import base64
import io
import zipfile

import pandas as pd
import pytest

from app.history_import import ImportFileError, looks_like_episode, preview

ALICE = {"X-User-Id": "alice"}

# a small catalogue with the hard cases: remakes, an IMDb id, an alternative title, a near-duplicate name
ITEMS = pd.DataFrame(
    {
        "movie_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "title": [
            "Inception",
            "Dune",
            "Dune",
            "RRR",
            "RRRrrrr!!!",
            "Star Wars: Episode VII - The Force Awakens",
            "Spirited Away (Sen to Chihiro no kamikakushi)",
            "The Dark Knight",
        ],
        "year": pd.array([2010, 1984, 2021, 2022, 2004, 2015, 2001, 2008], dtype="Int16"),
        "popularity": [900, 100, 500, 300, 5, 800, 400, 950],
        "imdb_id": [1375666, 87182, 1160419, 8178634, 377818, 2488496, 245429, 468569],
    }
)


def run(text: str, name: str = "NetflixViewingHistory.csv"):
    return preview(name, text, "text", ITEMS)


def test_netflix_history_matches_films_and_skips_tv():
    p = run(
        '"Title","Date"\n'
        '"Stranger Things: Season 4: Chapter One: The Hellfire Club","21/11/2026"\n'
        '"RRR (Hindi)","20/11/2026"\n'
        '"Star Wars: The Force Awakens","15/11/2026"\n'
        '"Inception","14/11/2026"\n'
        '"Dune","10/11/2026"\n'
        '"Inception","25/10/2026"\n'
        '"Money Heist: Part 3: We Are Back","20/10/2026"\n'
        '"Not A Film We Have","01/10/2026"\n'
    )
    assert p.source == "netflix" and p.rows == 8
    got = {m.movie_id: m for m in p.matches}
    assert set(got) == {4, 6, 1, 3}  # RRR (not RRRrrrr!!!), Episode VII, Inception, Dune 2021
    assert got[1].times == 2 and got[1].at.day == 14  # a rewatch: counted twice, the latest date kept
    assert all(m.kind == "watched" and m.rating is None for m in p.matches)
    assert p.tv_skipped == 2 and p.unmatched == ["Not A Film We Have"]
    assert [m.movie_id for m in p.matches] == [4, 6, 1, 3]  # newest first
    assert p.matches[0].at.month == 11 and p.matches[0].at.day == 20  # read as day/month/year


def test_netflix_dates_follow_the_files_order():
    p = run('"Title","Date"\n"Inception","3/4/21"\n"The Dark Knight","1/15/21"\n')  # US month/day/year
    dates = {m.movie_id: m.at for m in p.matches}
    assert (dates[1].month, dates[1].day) == (3, 4) and (dates[8].month, dates[8].day) == (1, 15)


def test_a_remake_resolves_to_the_film_released_before_it_was_watched():
    p = run('"Title","Date"\n"Dune","05/06/2000"\n')
    assert [m.movie_id for m in p.matches] == [2]  # in 2000 only the 1984 film existed


def test_full_netflix_download_skips_trailers_and_brief_views():
    p = run(
        "Profile Name,Start Time,Duration,Attributes,Title,Supplemental Video Type,Device Type\n"
        "Asha,2026-11-20 20:00:00,02:10:00,,Inception,,TV\n"
        "Asha,2026-11-19 20:00:00,00:02:00,,Dune: Trailer 2,TRAILER,TV\n"
        "Asha,2026-11-18 20:00:00,00:04:00,,The Dark Knight,,TV\n"
        "Asha,2026-11-17 20:00:00,00:45:00,,RRR (Telugu),,TV\n"
        "Asha,2026-11-17 21:00:00,00:50:00,,RRR (Telugu),,TV\n",
        name="ViewingActivity.csv",
    )
    assert {m.movie_id for m in p.matches} == {1, 4}
    assert p.short_skipped == 1  # four minutes of The Dark Knight is not a viewing
    assert next(m for m in p.matches if m.movie_id == 4).at.hour == 21  # the latest session


def test_imdb_ratings_match_by_id_and_convert_to_half_stars():
    p = run(
        "Const,Your Rating,Date Rated,Title,Original Title,URL,Title Type,IMDb Rating,Runtime (mins),Year\n"
        "tt1375666,9,2024-02-01,Inception (renamed),Inception,u,Movie,8.8,148,2010\n"
        "tt0468569,6,2024-01-01,The Dark Knight,The Dark Knight,u,Movie,9.0,152,2008\n"
        "tt4574334,10,2024-01-01,Stranger Things,Stranger Things,u,TV Series,8.7,51,2016\n",
        name="ratings.csv",
    )
    got = {m.movie_id: m for m in p.matches}
    assert p.source == "imdb" and p.tv_skipped == 1
    assert got[1].kind == "rating" and got[1].rating == 4.5  # matched by IMDb id despite the title
    assert got[8].rating == 3.0


def test_a_malformed_imdb_id_in_the_catalogue_is_ignored():
    """New films store IMDb numbers as text, and source data can hold a person's id ("nm..."): the
    catalogue skips what is not a title number instead of failing every import."""
    items = pd.concat(
        [
            ITEMS,
            pd.DataFrame(
                {
                    "movie_id": [1_000_000_001, 1_000_000_002],
                    "title": ["Vikram Vedha", "Some New Film"],
                    "year": pd.array([2017, 2025], dtype="Int16"),
                    "popularity": [0, 0],
                    "imdb_id": ["6148156", "nm7650962"],
                }
            ),
        ],
        ignore_index=True,
    )
    p = preview(
        "ratings.csv",
        "Const,Your Rating,Date Rated,Title,Original Title,URL,Title Type,IMDb Rating,Runtime (mins),Year\n"
        "tt6148156,9,2024-02-01,Vikram Vedha,Vikram Vedha,u,Movie,8.2,147,2017\n",
        "text",
        items,
    )
    assert [m.movie_id for m in p.matches] == [1_000_000_001]  # digits as text still match by id


def test_imdb_watchlist_goes_to_my_list():
    p = run(
        "Position,Const,Created,Modified,Description,Title,URL,Title Type,Year\n"
        "1,tt0245429,2024-03-01,,,Spirited Away,u,Movie,2001\n"
    )
    assert [(m.movie_id, m.kind) for m in p.matches] == [(7, "watchlist")]


def zip_of(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(f"letterboxd-asha-2026/{name}", text)
    return base64.b64encode(buf.getvalue()).decode()


def test_letterboxd_export_zip():
    content = zip_of(
        {
            "ratings.csv": "Date,Name,Year,Letterboxd URI,Rating\n2026-01-10,Dune,2021,u,4.5\n",
            "watched.csv": "Date,Name,Year,Letterboxd URI\n"
            "2026-01-09,Dune,2021,u\n2025-12-01,Sen to Chihiro no kamikakushi,2001,u\n",
            "watchlist.csv": "Date,Name,Year,Letterboxd URI\n2026-02-01,Inception,2010,u\n"
            "2026-02-01,Dune,2021,u\n",
            "profile.csv": "Username\nasha\n",
        }
    )
    p = preview("letterboxd-asha.zip", content, "base64", ITEMS)
    got = {m.movie_id: m for m in p.matches}
    assert p.source == "letterboxd"
    assert (
        got[3].kind == "rating" and got[3].rating == 4.5 and got[3].times == 2
    )  # rated beats watched and My list
    assert got[3].at.month == 1  # the viewing date, not the date it was put on My list
    assert got[7].kind == "watched"  # matched through the alternative title
    assert got[1].kind == "watchlist"


def test_letterboxd_year_must_agree():
    p = run("Date,Name,Year,Letterboxd URI\n2026-01-09,Dune,1995,u\n", name="watched.csv")
    assert p.matches == [] and p.unmatched == ["Dune"]  # no Dune near 1995: better unmatched than wrong


@pytest.mark.parametrize(
    ("name", "content", "encoding", "message"),
    [
        ("x.csv", "Film,Stars\nInception,5\n", "text", "not a Netflix, IMDb or Letterboxd export"),
        ("x.zip", "not base64!", "base64", "could not be read"),
        ("x.zip", zip_of({"notes.txt": "hi"}), "base64", "no Letterboxd files"),
        ("x.csv", "Title,Date\n" + "x" * (11 * 1024 * 1024), "text", "larger than 10 MB"),
    ],
)
def test_unreadable_files_say_why(name, content, encoding, message):
    with pytest.raises(ImportFileError, match=message):
        preview(name, content, encoding, ITEMS)


def test_episode_titles():
    assert looks_like_episode("Wednesday: Season 1: Wednesday's Child Is Full of Woe")
    assert looks_like_episode("Narcos: Mexico: Season 2")
    assert looks_like_episode("Our Planet: Limited Series")
    assert not looks_like_episode("Spider-Man: No Way Home")


# --------------------------------------------------------------------------------------- through the API
# The tiny test bundle has films "Movie 0".."Movie 19" (ids 100-119); viewers like either 0-9 or 10-19.


def netflix(*titles: str) -> dict:
    rows = "".join(f'"{t}","11/{20 - i}/26"\n' for i, t in enumerate(titles))
    return {"name": "NetflixViewingHistory.csv", "content": '"Title","Date"\n' + rows}


def test_import_preview_then_save(client):
    p = client.post(
        "/me/import/preview",
        json=netflix("Movie 1", "Movie 2", "Movie 3", "Show: Season 1: Pilot", "Elsewhere"),
        headers=ALICE,
    )
    assert p.status_code == 200, p.text
    body = p.json()
    assert [i["movie"]["movie_id"] for i in body["items"]] == [101, 102, 103]
    assert (
        body["tv_skipped"] == 1
        and body["unmatched_count"] == 1
        and not any(i["already"] for i in body["items"])
    )

    items = [{"movie_id": i["movie"]["movie_id"], "kind": i["kind"], "at": i["at"]} for i in body["items"]]
    r = client.post("/me/import", json={"source": "netflix", "items": items}, headers=ALICE)
    assert r.json() == {"ratings": 0, "watched": 3, "watchlist": 0, "unchanged": 0}

    profile = client.get("/me/profile", headers=ALICE).json()
    assert profile["n_watched"] == 3 and profile["n_ratings"] == 0 and profile["onboarded"]
    recs = [r["movie_id"] for r in client.get("/recommendations", params={"k": 5}, headers=ALICE).json()]
    assert not {101, 102, 103} & set(recs)  # watched films are not recommended again
    assert sum(100 <= m <= 109 for m in recs) >= 4  # and they steer the picks to the same group
    watched = client.get("/me/watched", headers=ALICE).json()
    assert [w["movie_id"] for w in watched] == [101, 102, 103] and watched[0]["source"] == "netflix"

    again = client.post("/me/import/preview", json=netflix("Movie 1"), headers=ALICE).json()
    assert again["items"][0]["already"]  # the preview marks what is already in Red Box


def test_import_never_overwrites_a_rating_and_low_ratings_win(client):
    who = {"X-User-Id": "imran"}
    client.put("/ratings/105", json={"rating": 1}, headers=who)
    items = [
        {"movie_id": 105, "kind": "rating", "rating": 5, "at": "2026-01-01T00:00:00Z"},
        {"movie_id": 106, "kind": "rating", "rating": 4.5, "at": "2026-01-02T00:00:00Z"},
        {"movie_id": 107, "kind": "watchlist"},
    ]
    r = client.post("/me/import", json={"source": "imdb", "items": items}, headers=who).json()
    assert r == {"ratings": 1, "watched": 0, "watchlist": 1, "unchanged": 1}
    mine = {x["movie_id"]: x["rating"] for x in client.get("/me/ratings", headers=who).json()}
    assert mine == {105: 1, 106: 4.5}  # the app rating stays
    assert [m["movie_id"] for m in client.get("/me/watchlist", headers=who).json()] == [107]
    # a film imported as watched but rated low is not used as a liked film
    client.post(
        "/me/import", json={"source": "netflix", "items": [{"movie_id": 108, "kind": "watched"}]}, headers=who
    )
    client.put("/ratings/108", json={"rating": 1.5}, headers=who)
    recs = {r["movie_id"] for r in client.get("/recommendations", params={"k": 10}, headers=who).json()}
    assert 108 not in recs


def test_import_validation(client):
    bad = client.post("/me/import/preview", json={"name": "x.csv", "content": "a,b\n1,2\n"}, headers=ALICE)
    assert bad.status_code == 422 and "not a Netflix" in bad.json()["detail"]
    unknown = {"source": "netflix", "items": [{"movie_id": 999, "kind": "watched"}]}
    assert client.post("/me/import", json=unknown, headers=ALICE).status_code == 404
    no_stars = {"source": "imdb", "items": [{"movie_id": 100, "kind": "rating"}]}
    assert client.post("/me/import", json=no_stars, headers=ALICE).status_code == 422
    half = {"source": "imdb", "items": [{"movie_id": 100, "kind": "rating", "rating": 3.3}]}
    assert client.post("/me/import", json=half, headers=ALICE).status_code == 422


def test_imported_history_can_be_removed_exported_and_deleted(client):
    who = {"X-User-Id": "zoe"}
    items = [{"movie_id": m, "kind": "watched"} for m in (110, 111, 112)]
    client.post("/me/import", json={"source": "letterboxd", "items": items}, headers=who)
    assert client.delete("/me/watched/110", headers=who).status_code == 204
    assert {w["movie_id"] for w in client.get("/me/watched", headers=who).json()} == {111, 112}
    exported = client.get("/me/export", headers=who).json()
    assert {w["movie_id"] for w in exported["imported_history"]} == {111, 112}
    assert client.delete("/me/watched", headers=who).status_code == 204
    assert client.get("/me/watched", headers=who).json() == []
    client.post("/me/import", json={"source": "letterboxd", "items": items}, headers=who)
    assert client.post("/me/delete", json={"password": ""}, headers=who).status_code == 204
    assert client.get("/me/watched", headers=who).json() == []
