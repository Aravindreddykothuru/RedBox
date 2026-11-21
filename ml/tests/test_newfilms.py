"""New films (item cold start): text vectors mapped to iALS factors, served after the model's films."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.serving.engine import Engine
from recsys.serving.newfilms import (
    ALPHAS,
    NEW_ID_OFFSET,
    choose_alpha,
    fit_mapping,
    fold_in,
    known_imdb_ids,
    predict_factors,
)
from recsys.serving.scorers import IALSScorer
from recsys.serving.testing import make_tiny_bundle


def test_mapping_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 12))
    A = rng.normal(size=(12, 4))
    Q = X @ A + 0.5 + 0.01 * rng.normal(size=(400, 4))
    alpha, r2 = choose_alpha(X, Q)
    assert alpha in ALPHAS and r2[alpha] > 0.99
    W = fit_mapping(X[:300], Q[:300], alpha)
    assert np.abs(predict_factors(W, X[300:]) - Q[300:]).max() < 0.1


def test_fold_in_matches_the_ials_scorer():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(30, 5))
    scorer = IALSScorer(Q, alpha=4.0, reg=2.0)
    scorer.set_history(
        sp.csr_matrix(([1.0, 1.0, 1.0], ([0, 0, 0], [2, 7, 11])), shape=(1, 30)), [np.array([2, 7, 11])]
    )
    expected = scorer.user_vectors(np.array([0]))[0]
    assert np.allclose(fold_in(Q, np.array([2, 7, 11]), 4.0, scorer.gram), expected)


def test_new_films_are_served_after_the_models_films(tmp_path):
    path = make_tiny_bundle(tmp_path / "tiny")
    base = Engine(path)
    n, dim = base.n_items, base.text_emb.shape[1]
    new = pd.DataFrame(
        {
            "movie_id": [NEW_ID_OFFSET + 1, NEW_ID_OFFSET + 2],
            "title": ["New One", "New Two"],
            "year": pd.array([2025, 2026], dtype="Int16"),
            "genres": [["Action"], ["Drama"]],
            "popularity": 0,
            "imdb_id": ["1234567", None],
            "poster": ["https://example.org/1.jpg", None],
            "overview": ["A story.", "Another story."],
            "overview_source": "wikipedia",
            "language": ["Telugu", "Tamil"],
            "keywords": [[], []],
        }
    )
    new.to_parquet(path / "new_items.parquet", index=False)
    np.save(path / "new_items_factors.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(path / "new_items_text.npy", np.eye(2, dim, dtype=np.float32))

    e = Engine(path)
    assert e.n_items == n and len(e.items) == n + 2  # the models' films are unchanged
    assert e.movie_to_idx[NEW_ID_OFFSET + 2] == n + 1 and e.new_languages == ["Telugu", "Tamil"]
    assert 1234567 not in known_imdb_ids(e)  # a rebuild replaces the new films, it doesn't skip them as known
    assert e.new_films([0, 1]) == []  # no iALS generator in the tiny bundle: no row
    e.scorers["ials"] = IALSScorer(np.ones((n, 2)) * np.array([1.0, 0.1]), alpha=1.0, reg=1.0)
    picks = e.new_films([0, 1], k=5)
    assert [p.movie_id for p in picks] == [NEW_ID_OFFSET + 1, NEW_ID_OFFSET + 2]  # the user leans to factor 0
    assert [p.reason for p in picks] == [
        "New in Telugu: its story matches your taste",
        "New in Tamil: its story matches your taste",
    ]
    assert [p.movie_id for p in e.new_films([0, 1], language="Tamil")] == [NEW_ID_OFFSET + 2]
    assert e.new_films([0, 1], language="Kannada") == []  # a language without new films: nothing
    assert [p.movie_id for p in e.new_films([0, 1], exclude={n})] == [NEW_ID_OFFSET + 2]
    del e.scorers["ials"]  # the stand-in is not one of the tiny ranker's generators
    assert e.why(n, [0, 1]) is None  # the ranker never scores new films
    assert e.similar(n, k=3, mode="cf")[0].reason == "Similar story and themes"  # no co-ratings yet
    assert e.recommend([0, 1], exclude={n, n + 1}, k=3)  # exclusions may name new films safely
    e.items["avg_rating"] = 4.0  # a real bundle has rating statistics for its films (not for new ones)
    e.hidden_gems([0, 1], k=3)  # model-sized arrays only: no shape clash with the longer catalogue


def test_describes_film_keeps_films_and_drops_what_an_article_redirects_to():
    from recsys.data.wikipedia import describes_film

    film = "Khaidi No. 150 is a 2017 Indian Telugu-language action comedy film directed by V. V. Vinayak."
    assert describes_film(film, 2017)
    assert describes_film("Lens is 2016 Indian drama thriller film written by J. Radhakrishnan.", 2016)
    assert describes_film("Sathamanam Bhavati is a 2017 Indian Telugu-language family drama.", 2017)
    # released later than Wikidata's date: the same film, delayed
    assert describes_film("Saajan Bakery Since 1962 is a 2021 Indian Malayalam-language film.", 2019)
    assert not describes_film("Venu Sriram is an Indian film director and screenwriter.", 2016)  # a person
    assert not describes_film("Vinay Rai (born 18 September 1979) is an Indian actor in Tamil films.", 2017)
    assert not describes_film("AVM Productions is an Indian film production studio.", 2018)  # a company
    assert not describes_film("This is a list of films released in 2017.", 2017)
    assert not describes_film("Arabia Kadali is an Indian Telugu survival drama television series.", 2025)
    assert not describes_film("The Love Guru is a 2008 romantic comedy film.", 2016)  # an older film
    assert not describes_film("India receives an average annual precipitation of 1,170 mm.", 2019)
    assert not describes_film(None, 2019)


def test_usable_films_filters_and_keeps_each_film_once():
    from recsys.serving.newfilms import usable_films

    master = "Master is a 2021 Indian Tamil-language action drama film directed by Lokesh Kanagaraj."
    films = pd.DataFrame(
        {
            "wikidata": ["Q1", "Q2", "Q3", "Q4", "Q4", "Q5"],
            "imdb": ["tt0000001", "nm7650962", None, "tt0000004", "tt0000004", "tt0000005"],
            "title": ["Thalapathy 64", "Master", "Shukra", "Bilingual", "Bilingual", "Known"],
            "year": [2020, 2021, 2021, 2019, 2019, 2018],
            "summary": [
                master,
                master,
                "Arvind Krishna (born 5 January 1985) is an Indian actor.",
                "Bilingual is a 2019 Indian Tamil and Telugu-language thriller film.",
                "Bilingual is a 2019 Indian Tamil and Telugu-language thriller film.",
                "Known is a 2018 Indian Malayalam-language film.",
            ],
            "language": ["Tamil", "Tamil", "Telugu", "Tamil", "Telugu", "Malayalam"],
        }
    )
    out = usable_films(films, known_imdb={5})
    assert out["title"].tolist() == ["Master", "Bilingual"]  # the article's own title; the first language
    assert out.loc[out["title"] == "Bilingual", "language"].item() == "Tamil"
    assert out.loc[out["title"] == "Master", "imdb"].isna().all()  # a person's IMDb id is not kept


def test_warm_factors_move_from_the_story_to_the_raters():
    from recsys.serving.newfilms import warm_factors

    prior = np.array([0.0, 1.0])
    assert np.allclose(warm_factors(prior, np.zeros((0, 2)), np.zeros(0), 5.0, 10.0), prior)  # no ratings
    likers = np.tile([1.0, 0.0], (50, 1))  # viewers who all lean to factor 0
    many = warm_factors(prior, likers, np.ones(50), 5.0, 10.0)
    few = warm_factors(prior, likers[:1], np.ones(1), 5.0, 10.0)
    assert many[0] > few[0] > prior[0]  # each like pulls it further towards its likers' taste
    assert many @ [1.0, 0.0] > 0.9  # with many likes they would all be recommended it
    strong = warm_factors(prior, likers[:1], np.ones(1), 5.0, 1000.0)
    assert np.abs(strong - prior).max() < 0.01  # a strong prior: one rating barely moves it
    disliked = warm_factors(prior, likers[:5], np.zeros(5), 5.0, 10.0)
    assert disliked[0] <= prior[0]  # low ratings push it away from those viewers


def _engine_with_new_films(tmp_path, n_new: int = 3) -> Engine:
    path = make_tiny_bundle(tmp_path / "tiny")
    n, dim = Engine(path).n_items, None
    dim = Engine(path).text_emb.shape[1]
    langs = ["English", "Tamil", "English", "Hindi"][:n_new]
    pd.DataFrame(
        {
            "movie_id": [NEW_ID_OFFSET + i for i in range(1, n_new + 1)],
            "title": [f"New {i}" for i in range(1, n_new + 1)],
            "year": pd.array([2024, 2025, 2026, 2025][:n_new], dtype="Int16"),
            "genres": [["Action"], ["Drama"], ["Comedy"], ["Drama"]][:n_new],
            "popularity": 0,
            "imdb_id": [None] * n_new,
            "poster": [None] * n_new,
            "overview": ["A story."] * n_new,
            "overview_source": "wikipedia",
            "language": langs,
            "links": [40, 3, 12, 5][:n_new],
            "keywords": [[] for _ in range(n_new)],
        }
    ).to_parquet(path / "new_items.parquet", index=False)
    np.save(path / "new_items_factors.npy", np.eye(n_new, 2, dtype=np.float32))
    np.save(path / "new_items_text.npy", np.zeros((n_new, dim), dtype=np.float32))
    e = Engine(path)
    e.scorers["ials"] = IALSScorer(np.tile([[1.0, 0.0], [0.0, 1.0]], (n // 2 + 1, 1))[:n], alpha=4.0, reg=1.0)
    assert e.n_items == n
    return e


def test_new_users_get_the_best_known_releases_and_their_genres(tmp_path):
    e = _engine_with_new_films(tmp_path)
    n = e.n_items
    assert e.new_languages == ["English", "Tamil"]
    plain = e.new_films([], k=3)
    assert [p.item_idx for p in plain] == [n, n + 2, n + 1]  # by how widely known (40, 12, 3 Wikipedias)
    assert plain[0].reason == "New in English: widely talked about"
    drama = e.new_films([], k=3, genres=["Drama"])
    assert drama[0].item_idx == n + 1 and drama[0].reason == "New in Tamil: matches the genres you picked"
    assert [p.item_idx for p in e.new_films([], language="Tamil")] == [n + 1]


def test_ratings_warm_up_a_new_film_and_releases_can_be_added_live(tmp_path):
    e = _engine_with_new_films(tmp_path)
    n = e.n_items
    leans_1 = [i for i in range(n) if i % 2 == 1][:3]  # viewers whose taste is factor 1
    before = e.new_films(leans_1, k=3)
    assert before[0].item_idx == n + 1  # its story's factors (0, 1) match them

    def score_of_n() -> float:
        return next(r.score for r in e.new_films(leans_1, k=3) if r.item_idx == n)

    # viewers like them like film n (story factors (1, 0)): its factors move towards their taste, more
    # with each like, until it is their top new film
    assert score_of_n() == 0.0
    e.set_feedback({n: [(leans_1, True)] * 5})
    five = score_of_n()
    e.set_feedback({n: [(leans_1, True)] * 30})
    after = e.new_films(leans_1, k=3)
    assert 0.0 < five < after[0].score and after[0].item_idx == n
    assert after[0].reason == "New in English: liked by 30 Red Box viewers"
    assert e.new_likes[0] == 30 and np.allclose(e.new_prior[0], [1.0, 0.0])  # the prior is kept
    e.set_feedback({})  # feedback is recomputed from scratch each time
    assert e.new_films(leans_1, k=1)[0].item_idx == n + 1

    rows = pd.DataFrame(
        {
            "movie_id": [NEW_ID_OFFSET + 9],
            "title": ["Fresh"],
            "year": pd.array([2026], dtype="Int16"),
            "genres": [["Drama"]],
            "language": ["Hindi"],
            "links": [7],
        }
    )
    e.add_new_films(rows, np.array([[0.0, 2.0]]), np.zeros((1, e.text_emb.shape[1])))
    assert e.movie_to_idx[NEW_ID_OFFSET + 9] == n + 3 and e.movie_to_idx[NEW_ID_OFFSET + 1] == n
    assert e.new_languages == ["English", "Tamil", "Hindi"]
    assert e.new_films(leans_1, k=1)[0].item_idx == n + 3  # the strongest match, served at once
    assert len(e.text_emb) == len(e.items)


def test_write_new_items_round_trips(tmp_path):
    from recsys.serving.newfilms import write_new_items

    e = _engine_with_new_films(tmp_path)
    e.add_new_films(
        pd.DataFrame(
            {
                "movie_id": [NEW_ID_OFFSET + 9],
                "title": ["Fresh"],
                "year": pd.array([2026], dtype="Int16"),
                "genres": [["Drama"]],
                "language": ["Hindi"],
                "links": [7],
            }
        ),
        np.array([[0.0, 2.0]]),
        np.zeros((1, e.text_emb.shape[1])),
    )
    write_new_items(e)
    again = Engine(e.bundle)
    assert again.items["movie_id"].iloc[again.n_items :].tolist() == [NEW_ID_OFFSET + i for i in (1, 2, 3, 9)]
    assert np.allclose(again.new_prior[-1], [0.0, 2.0]) and len(again.text_emb) == len(again.items)


def test_parallel_requests_are_scored_with_their_own_history(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    e = Engine(make_tiny_bundle(tmp_path / "tiny"))
    a, b = [0, 1, 2], [5, 6, 7]
    expected = {
        tuple(a): [r.item_idx for r in e.recommend(a, k=5)],
        tuple(b): [r.item_idx for r in e.recommend(b, k=5)],
    }
    with ThreadPoolExecutor(8) as pool:
        jobs = [(h, pool.submit(lambda h=h: [r.item_idx for r in e.recommend(h, k=5)])) for h in [a, b] * 40]
        assert all(f.result() == expected[tuple(h)] for h, f in jobs)


def test_a_new_film_liked_in_the_app_graduates_into_the_catalogue(tmp_path):
    from datetime import UTC, datetime

    from test_movielens import _ml32m_dir

    from recsys.data.movielens import load_ml32m

    t2020 = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    new_id = NEW_ID_OFFSET + 4242
    rows = [(f"k-{u}", m, 5.0, t2020 + m) for u in range(3) for m in (10, 11, 12)]
    rows += [(f"k-{u}", new_id, 5.0, t2020 + 99) for u in range(3)]  # three app users liked a new film
    rows += [("k-0", NEW_ID_OFFSET + 7, 5.0, t2020)]  # a film nobody else liked, unknown anyway
    app_file = tmp_path / "app.csv"
    pd.DataFrame(rows, columns=["user_key", "movie_id", "rating", "timestamp"]).to_csv(app_file, index=False)
    films = pd.DataFrame(
        {"movie_id": [new_id], "title": ["Leo"], "year": [2023], "genres": [["Action", "Thriller"]]}
    )
    ds = load_ml32m(
        _ml32m_dir(tmp_path), since="2018-01-01", min_user=3, min_item=3, app_file=app_file, new_films=films
    )
    items = ds.items.set_index("item_id")
    assert items.loc[new_id, "title"] == "Leo" and items.loc[new_id, "genres"] == ["Action", "Thriller"]
    assert items.loc[new_id, "year"] == 2023 and ds.meta["new_films_graduated"] == 1
    assert (ds.interactions["item_id"] == new_id).sum() == 3


def test_a_daily_look_that_finds_nothing_new_changes_nothing(tmp_path):
    from recsys.serving.newfilms import add_releases, usable_films

    e = _engine_with_new_films(tmp_path)
    known = pd.DataFrame(
        {
            "wikidata": ["Q1", "Q2"],  # the films already here
            "imdb": [None, None],
            "title": ["New 1", "New 2"],
            "year": [2024, 2025],
            "genres": [["Action"], ["Drama"]],
            "summary": ["New 1 is a 2024 action film.", "New 2 is a 2025 drama film."],
            "poster_url": [None, None],
            "language": ["English", "Tamil"],
            "links": [40, 3],
        }
    )
    before = len(e.items)
    assert add_releases(e, known, encoder=object()).empty and len(e.items) == before
    nobody = known.assign(summary=["Someone (born 1990) is an actor.", "A list of films."])
    assert usable_films(nobody, set()).empty  # every row filtered out: still a table with its columns
    assert list(usable_films(nobody, set()).columns) == list(usable_films(known, set()).columns)
