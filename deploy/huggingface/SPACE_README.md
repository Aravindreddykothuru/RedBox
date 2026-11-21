---
title: Red Box API
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid collaborative-filtering film recommender (FastAPI)
---

# Red Box API

The recommendation service behind Red Box: film recommendations by collaborative filtering. Five models
(EASE, SASRec, iALS, a Markov chain and popularity) propose films, and a LightGBM ranker orders them.

- Interactive docs: `/docs`
- Health and model info: `/health`
- New releases are added every day from Wikidata and Wikipedia. They are placed in the models' taste
  space from their story, then learn from the app's own ratings.

**Credits.** Ratings data: MovieLens 32M (GroupLens Research). Viewer keywords: MovieLens 32M tags.
Summaries: Wikipedia (CC BY-SA), and film data from Wikidata (CC0). Posters are the studios' artwork, as
shown on Wikipedia. Sentence encoder: all-MiniLM-L6-v2 (Apache 2.0). The raw MovieLens viewing histories
are not included; the models and aggregate statistics are.
