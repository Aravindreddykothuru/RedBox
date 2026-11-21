from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.movielens import load_ml1m, split_title
from recsys.data.schema import RecDataset, SchemaError


@pytest.mark.parametrize(
    ("raw", "title", "year"),
    [
        ("Toy Story (1995)", "Toy Story", 1995),
        ("Usual Suspects, The (1995)", "The Usual Suspects", 1995),
        ("Misérables, Les (1995)", "Les Misérables", 1995),
        ("Slingshot, The (Kådisbellan ) (1993)", "The Slingshot (Kådisbellan )", 1993),
        ("Shawshank Redemption, The (1994)", "The Shawshank Redemption", 1994),
        ("No Year Here", "No Year Here", None),
    ],
)
def test_split_title(raw, title, year):
    assert split_title(raw) == (title, year)


def test_load_ml1m_parses_and_types(ml1m_dir):
    ds = load_ml1m(ml1m_dir)
    inter = ds.interactions
    assert len(inter) == 6
    assert inter.dtypes.astype(str).to_dict() == {
        "user_id": "int32",
        "item_id": "int32",
        "rating": "float32",
        "timestamp": "int64",
    }
    assert len(ds.items) == 4  # catalog keeps never-rated movies
    les_mis = ds.items.set_index("item_id").loc[20]
    assert les_mis["title"] == "Les Misérables"  # Latin-1 decoded correctly
    assert les_mis["genres"] == ["Drama", "Musical"]
    assert ds.users.set_index("user_id").loc[2, "age_group"] == "56+"
    assert ds.users.set_index("user_id").loc[1, "occupation"] == "K-12 student"


def test_save_load_roundtrip(ml1m_dir, tmp_path):
    ds = load_ml1m(ml1m_dir)
    ds.save(tmp_path / "out")
    back = RecDataset.load(tmp_path / "out")
    pd.testing.assert_frame_equal(back.interactions, ds.interactions)
    assert back.items["genres"].tolist() == ds.items["genres"].tolist()
    assert back.meta["rating_scale"] == [1, 5]


def test_validate_rejects_duplicate_pairs(ml1m_dir):
    ds = load_ml1m(ml1m_dir)
    ds.interactions = pd.concat([ds.interactions, ds.interactions.head(1)])
    with pytest.raises(SchemaError, match="duplicate"):
        ds.validate()


def test_validate_rejects_unknown_items(ml1m_dir):
    ds = load_ml1m(ml1m_dir)
    ds.interactions.loc[0, "item_id"] = 999
    with pytest.raises(SchemaError, match="unknown item_ids"):
        ds.validate()


def _ml32m_dir(tmp_path):
    """Tiny ML-32M-style folder: 2017 ratings (outside the window), 2019 ratings (inside)."""
    from datetime import UTC, datetime

    t2017 = int(datetime(2017, 6, 1, tzinfo=UTC).timestamp())
    t2019 = int(datetime(2019, 6, 1, tzinfo=UTC).timestamp())
    rows = []
    for u in range(1, 7):  # users 1-6 like films 10, 11, 12 in 2019 (3 likes each)
        for k, m in enumerate((10, 11, 12)):
            rows.append((u, m, 4.5, t2019 + 60 * k))
        rows.append((u, 13, 2.0, t2019 + 500))  # a disliked rating is kept for kept users and films
    rows.append((7, 10, 5.0, t2019))  # user 7 likes one film only -> dropped (min_user)
    rows.append((1, 14, 5.0, t2019 + 900))  # film 14 liked by one user -> dropped (min_item)
    rows += [(u, 15, 5.0, t2017) for u in range(1, 7)]  # film 15 only liked before the window
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        tmp_path / "ratings.csv", index=False
    )
    pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13, 14, 15],
            "title": [
                "Up (2009)",
                "Matrix, The (1999)",
                "Coco (2017)",
                "Film, A (2020)",
                "Solo (2018)",
                "Old (1950)",
            ],
            "genres": [
                "Animation|Children|IMAX",
                "Action|Sci-Fi",
                "(no genres listed)",
                "Drama",
                "Drama",
                "War",
            ],
        }
    ).to_csv(tmp_path / "movies.csv", index=False)
    return tmp_path


def test_load_ml32m_window_kcore_and_genres(tmp_path):
    from recsys.data.movielens import load_ml32m

    ds = load_ml32m(_ml32m_dir(tmp_path), since="2018-01-01", min_user=3, min_item=3)
    inter = ds.interactions
    assert sorted(inter["user_id"].unique()) == [1, 2, 3, 4, 5, 6]  # user 7 too few likes
    # films need min_item *likes*: 13 (only 2-star ratings) and 14 (one like) are dropped, and 15 was
    # liked only before the window
    assert sorted(inter["item_id"].unique()) == [10, 11, 12]
    assert inter["timestamp"].min() >= 1514764800  # 2018-01-01
    items = ds.items.set_index("item_id")
    assert items.loc[10, "genres"] == ["Animation", "Children's"]  # renamed, IMAX dropped
    assert items.loc[12, "genres"] == []
    assert items.loc[11, "title"] == "The Matrix" and items.loc[11, "year"] == 1999
    assert ds.meta["window_since"] == "2018-01-01"


def test_ml32m_merges_imported_app_ratings(tmp_path):
    from datetime import UTC, datetime

    from recsys.data.movielens import APP_USER_OFFSET, load_ml32m

    t2020 = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    app_file = tmp_path / "app.csv"
    rows = [("k-anna", m, 5.0, t2020 + m) for m in (10, 11, 12)]  # an app user who liked 3 kept films
    rows += [("k-ben", 10, 4.0, 1)]  # an app rating from 1970: outside the window
    pd.DataFrame(rows, columns=["user_key", "movie_id", "rating", "timestamp"]).to_csv(app_file, index=False)
    ds = load_ml32m(_ml32m_dir(tmp_path), since="2018-01-01", min_user=3, min_item=3, app_file=app_file)
    app_users = ds.interactions[ds.interactions["user_id"] >= APP_USER_OFFSET]
    assert len(set(app_users["user_id"])) == 1 and sorted(app_users["item_id"]) == [10, 11, 12]
    assert ds.meta["app_ratings_imported"] == 3
