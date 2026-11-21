from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data.schema import RecDataset
from recsys.data.split import Split, SplitConfig, SplitError, make_split


def _dataset(rows) -> RecDataset:
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"]).astype(
        {"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"}
    )
    item_ids = np.unique(inter["item_id"]).astype("int32")
    items = pd.DataFrame({"item_id": item_ids, "title": "x", "genres": [[] for _ in item_ids]})
    return RecDataset(name="toy", interactions=inter, items=items)


def _rows():
    rows = []
    rows += [(1, i, 5, 100 + i) for i in range(1, 11)]  # user 1: 10 liked, distinct times
    rows += [(1, 11, 2, 200)]  # user 1: disliked -> dropped at threshold 4
    rows += [(2, 1, 4, 100), (2, 2, 4, 101)]  # user 2: only 2 liked -> train-only
    rows += [(3, i, 5, 500) for i in range(1, 6)]  # user 3: 5 liked in the SAME second
    rows += [(4, i, 5, 100 + i) for i in range(1, 5)]  # user 4: 4 liked ...
    rows += [(4, 99, 5, 105)]  # ... newest is item 99 that nobody else has -> cold
    rows += [(5, 1, 1, 100), (5, 2, 2, 101)]  # user 5: no liked items -> dropped
    # user 7 likes items 10..1 in reverse order, so items 9 and 10 also occur in someone's TRAIN
    # (otherwise they would be cold items and correctly dropped from user 1's val/test)
    rows += [(7, 11 - t, 5, t) for t in range(1, 11)]
    return rows


def _items_of(split: Split, part: str, user: int) -> set[int]:
    uidx = int(np.flatnonzero(split.user_ids == user)[0])
    df = getattr(split, part)
    return set(split.item_ids[df.loc[df["user_idx"] == uidx, "item_idx"]].tolist())


@pytest.fixture
def split() -> Split:
    return make_split(_dataset(_rows()), SplitConfig())


def test_chronological_parts(split):
    assert _items_of(split, "train", 1) == set(range(1, 9))
    assert _items_of(split, "val", 1) == {9}
    assert _items_of(split, "test", 1) == {10}


def test_threshold_drops_disliked(split):
    n_disliked = sum(1 for r in _rows() if r[2] < 4)
    assert split.stats["n_kept_interactions"] == len(_rows()) - n_disliked
    assert 11 not in split.item_ids  # only ever disliked
    everything = make_split(_dataset(_rows()), SplitConfig(positive_threshold=None))
    assert everything.stats["n_kept_interactions"] == len(_rows())
    assert 5 in everything.user_ids  # a user with only low ratings is kept when every rating counts


def test_small_users_are_train_only(split):
    assert _items_of(split, "train", 2) == {1, 2}
    assert _items_of(split, "val", 2) == set()
    assert _items_of(split, "test", 2) == set()
    assert split.stats["n_train_only_users"] == 1


def test_users_without_positives_are_dropped(split):
    assert 5 not in split.user_ids
    assert split.stats["n_users_dropped_no_positives"] == 1


def test_cold_items_removed_from_test(split):
    assert 99 not in split.item_ids
    assert _items_of(split, "test", 4) == set()  # its only test item was cold
    assert split.stats["cold_item_interactions_dropped"]["test"] == 1


def test_same_second_ties_are_seeded_and_order_independent():
    rows = _rows()
    a = make_split(_dataset(rows), SplitConfig(seed=7))
    shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    b = make_split(_dataset(shuffled), SplitConfig(seed=7))
    assert a.fingerprint() == b.fingerprint()  # input file order does not matter
    outcomes = {
        frozenset(_items_of(make_split(_dataset(rows), SplitConfig(seed=s)), "test", 3)) for s in range(10)
    }
    assert len(outcomes) > 1  # the tie-break really is random across seeds...
    assert all(len(o) == 1 for o in outcomes)  # ...but always yields exactly one test item


def test_check_passes_and_catches_leakage(split):
    split.check()
    leaked = Split(
        **{**vars(split), "train": pd.concat([split.train, split.test.head(1)], ignore_index=True)}
    )
    with pytest.raises(SplitError, match="share user-item pairs"):
        leaked.check()
    future = split.train.copy()
    user1 = int(np.flatnonzero(split.user_ids == 1)[0])
    row = future.index[future["user_idx"] == user1][0]
    future.loc[row, "timestamp"] = 10**10  # a "training" event after user 1's test event
    with pytest.raises(SplitError, match="older than"):
        Split(**{**vars(split), "train": future}).check()


def test_save_load_roundtrip_and_tamper_detection(split, tmp_path):
    split.save(tmp_path / "s")
    back = Split.load(tmp_path / "s")
    assert back.fingerprint() == split.fingerprint()
    assert back.stats == split.stats
    tampered = back.test.copy()
    tampered.loc[0, "item_idx"] = (tampered.loc[0, "item_idx"] + 1) % back.n_items
    tampered.to_parquet(tmp_path / "s" / "test.parquet", index=False)
    with pytest.raises(SplitError, match="fingerprint"):
        Split.load(tmp_path / "s")


def test_matrices(split):
    trainval = split.matrix("train", "val")
    assert trainval.shape == (split.n_users, split.n_items)
    assert trainval.nnz == len(split.train) + len(split.val)
    data = split.train_data("train", "val")
    # chronological within each user: user 1's last train+val item is item 9
    uidx = int(np.flatnonzero(split.user_ids == 1)[0])
    last = data.interactions[data.interactions["user_idx"] == uidx]["item_idx"].iloc[-1]
    assert split.item_ids[last] == 9


def test_config_validation():
    with pytest.raises(ValueError):
        SplitConfig(val_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError):
        SplitConfig(min_positives=2)
    assert SplitConfig().name == "pos4_val0.1_test0.1_min3_seed42"


def test_saved_split_is_rebuilt_when_the_data_changes(tmp_path, monkeypatch):
    import os

    from recsys.experiments.runner import load_or_create_split
    from recsys.paths import processed_dir

    monkeypatch.setenv("RECSYS_DATA_DIR", str(tmp_path))
    _dataset(_rows()).save(processed_dir("toy"))
    first = load_or_create_split("toy", SplitConfig())
    again = load_or_create_split("toy", SplitConfig())
    assert again.stats["fingerprint"] == first.stats["fingerprint"]  # unchanged data: loaded, not rebuilt

    _dataset(_rows() + [(8, i, 5, 300 + i) for i in range(1, 6)]).save(processed_dir("toy"))  # new user
    future = os.path.getmtime(processed_dir("toy") / "interactions.parquet") + 10
    os.utime(processed_dir("toy") / "interactions.parquet", (future, future))
    rebuilt = load_or_create_split("toy", SplitConfig())
    assert 8 in rebuilt.user_ids and rebuilt.stats["fingerprint"] != first.stats["fingerprint"]
