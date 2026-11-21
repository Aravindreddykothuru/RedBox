"""Content side: viewer keywords, movie texts, the ONNX sentence encoder's pooling, semantic search."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recsys.content import build as build_mod
from recsys.content.build import add_content_to_bundle, item_text
from recsys.content.encoder import MINILM, TextEncoder
from recsys.content.keywords import movie_keywords, normalise_tag
from recsys.paths import models_dir
from recsys.serving.engine import Engine
from recsys.serving.testing import KeywordEncoder, make_tiny_bundle


def test_item_text_puts_keywords_before_the_plot():
    text = item_text("Alien", 1979, ["Horror", "Sci-Fi"], ["space", "creature"], "A crew...", None)
    assert text == "A 1979 film. Genres: Horror, Sci-Fi. Keywords: space, creature. A crew..."
    assert "Alien" not in text  # title words would outweigh the description in the embedding
    assert item_text("Untitled", None, [], []) == "Untitled."  # nothing else to go on


def test_movie_keywords_ranks_by_distinct_users_and_drops_noise():
    tags = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 1, 2, 4, 5, 5],
            "movieId": [10, 10, 10, 10, 10, 10, 20, 20, 20],
            "tag": ["Time Travel", "time  travel", "time travel", "dystopia", "dystopia", "DVD",
                    "quirky", "quirky", "dvd"],
        }
    )  # fmt: skip
    kw = movie_keywords(tags, [10, 20, 30])
    # "time travel": 3 users (case/space-normalised); "dystopia": 1 user, twice -> below min_users
    assert kw[10] == ["time travel"]
    # movie 20: no tag reaches 2 users -> fall back to its most-used tags; "dvd" is a stop tag
    assert kw[20] == ["quirky"]
    assert 30 not in kw
    assert normalise_tag("  Mind-Bending ") == "mind-bending"


def test_encoder_mean_pools_only_real_tokens_and_normalises():
    enc = object.__new__(TextEncoder)
    batch = [SimpleNamespace(ids=[5, 6, 0], attention_mask=[1, 1, 0])]
    enc.tokenizer = SimpleNamespace(encode_batch=lambda texts: batch[: len(texts)])
    hidden = np.array([[[3.0, 0.0], [1.0, 0.0], [100.0, 100.0]]], dtype=np.float32)  # pad token = noise
    enc.session = SimpleNamespace(run=lambda _, feeds: [hidden])
    enc._inputs = {"input_ids", "attention_mask"}
    v = enc.encode(["anything"])
    np.testing.assert_allclose(v, [[1.0, 0.0]], atol=1e-6)  # mean of real tokens (2, 0), unit length


def test_engine_semantic_search_and_content_similarity(tmp_path):
    engine = Engine(make_tiny_bundle(tmp_path / "b"), encoder=KeywordEncoder())
    assert engine.has_text
    hits = engine.semantic_search("a funny comedy please", k=5)
    assert len(hits) == 5 and all(h.genres == ["Comedy"] for h in hits)
    assert all(h.reason.startswith("Viewers tag it: comedy") for h in hits)
    similar = engine.similar(1, k=4, mode="content")  # movie 1 is a comedy
    assert all(s.genres == ["Comedy"] and s.item_idx != 1 for s in similar)
    assert similar[0].reason == "Similar story and themes"


def test_bundle_without_text_model_has_no_semantic_search(tmp_path):
    engine = Engine(make_tiny_bundle(tmp_path / "b"))  # no encoder, no text_model/ folder
    assert not engine.has_text
    with pytest.raises(ValueError, match="no text model"):
        engine.semantic_search("comedy")


def test_add_content_to_bundle_aligns_vectors_by_movie_id(tmp_path, monkeypatch):
    monkeypatch.setenv("RECSYS_DATA_DIR", str(tmp_path / "data"))
    bundle = make_tiny_bundle(tmp_path / "b")
    processed = tmp_path / "data" / "processed" / "tiny"
    processed.mkdir(parents=True)
    # content for movies 101 and 100 (reversed order) and an unknown movie 999
    pd.DataFrame(
        {"item_id": [101, 100, 999], "keywords": [["b"], ["a"], ["x"]], "text": ["t1", "t0", "tx"]}
    ).to_parquet(processed / "content.parquet")
    np.save(processed / "text_embeddings.npy", np.array([[0, 1], [1, 0], [5, 5]], dtype=np.float32))
    pd.DataFrame({"item_id": [100], "imdb_id": ["0114709"], "tmdb_id": [862]}).to_parquet(
        processed / "links.parquet"
    )
    pd.DataFrame({"item_id": [100, 100, 101], "rating": [5.0, 4.0, 2.0]}).to_parquet(
        processed / "interactions.parquet"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("model.onnx", "tokenizer.json"):
        (model_dir / name).write_text("stub")
    monkeypatch.setattr(build_mod, "fetch_text_model", lambda: model_dir)

    # Wikipedia posters and summaries for 100 and 101; a TMDB plot for 101 only
    wiki = {"poster_url": ["https://p/100.jpg", None], "summary": ["Wiki 100.", "Wiki 101."]}
    pd.DataFrame({"item_id": [100, 101], **wiki}).to_parquet(processed / "wiki.parquet")
    pd.DataFrame({"item_id": [101], "overview": ["TMDB 101."]}).to_parquet(processed / "tmdb.parquet")

    add_content_to_bundle(bundle, "tiny")
    info = add_content_to_bundle(bundle, "tiny")  # twice: a rerun must not relabel Wikipedia's summaries
    vectors = np.load(bundle / "text_embeddings.npy")
    np.testing.assert_array_equal(vectors[0], [1, 0])  # movie 100 is item 0
    np.testing.assert_array_equal(vectors[1], [0, 1])
    assert not vectors[2:].any()  # movies without content stay zero (never matched)
    items = pd.read_parquet(bundle / "items.parquet")
    assert list(items.loc[0, "keywords"]) == ["a"] and list(items.loc[5, "keywords"]) == []
    assert items.loc[0, "tmdb_id"] == 862
    assert items.loc[0, "avg_rating"] == 4.5 and items.loc[0, "n_ratings"] == 2
    assert items.loc[5, "n_ratings"] == 0
    assert items.loc[0, ["overview", "overview_source", "poster"]].tolist() == [
        "Wiki 100.",
        "wikipedia",
        "https://p/100.jpg",
    ]
    assert items.loc[1, ["overview", "overview_source"]].tolist() == ["TMDB 101.", "tmdb"]
    assert pd.isna(items.loc[5, "overview_source"])
    assert (bundle / "text_model" / "model.onnx").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["text_model"]["name"] == MINILM.name and info["items_with_keywords"] == 2


@pytest.mark.skipif(
    not (models_dir("text", MINILM.name) / "model.onnx").exists(), reason="MiniLM not downloaded"
)
def test_real_encoder_understands_meaning():
    enc = TextEncoder(models_dir("text", MINILM.name))
    q, space, romance = enc.encode(
        ["astronauts lost in outer space", "A science-fiction film about a spaceship crew",
         "A romantic comedy about a wedding in Paris"]
    )  # fmt: skip
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
    assert q @ space > q @ romance + 0.2


def test_hidden_gems_respect_popularity_band_and_quality(tmp_path):
    bundle = make_tiny_bundle(tmp_path / "b")
    items = pd.read_parquet(bundle / "items.parquet")
    pop = items["popularity"].to_numpy()
    lo, hi = np.percentile(pop, [25, 75])
    in_band = np.flatnonzero((pop >= lo) & (pop < hi))
    assert in_band.size >= 3  # the fixture must have films in the band for this test to mean anything
    items["avg_rating"] = np.float32(4.0)
    items.loc[in_band[0], "avg_rating"] = 2.0  # a poorly rated film is never a gem
    items.to_parquet(bundle / "items.parquet", index=False)
    engine = Engine(bundle, encoder=KeywordEncoder())
    history = [int(i) for i in in_band[1:3]]
    gems = engine.hidden_gems(history, k=20)
    chosen = {g.item_idx for g in gems}
    assert gems and chosen <= set(in_band.tolist()) - {int(in_band[0])} - set(history)
