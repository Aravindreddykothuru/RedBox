# Chapter 9: Implementation

This chapter describes how the system was built:
- **ML package:** the research pipeline and the model export;
- **API:** serving recommendations;
- **web application;**
- **deployment.**

All code is our own, written with free, open-source tools. The models are implemented from their papers;
no recommender library is used.

## 9.1 Technology stack

| Layer | Technology | Role |
|---|---|---|
| Language and environments | Python 3.12 with **uv** (lock files); Node.js 24 | reproducible installs |
| Data | pandas, PyArrow (Parquet), SciPy sparse matrices | interaction matrices, splits |
| Models | NumPy/SciPy (neighbourhood models, EASE, iALS, Markov chain); **PyTorch** on CPU (BPR-MF, NeuMF, Mult-VAE, LightGCN, SASRec) | the eleven algorithms |
| Ranking | **LightGBM** (LambdaRank) | stage 2 of the hybrid |
| Tuning and tracking | **Optuna** (TPE, median pruning); **MLflow** (SQLite store) | equal-budget tuning; every run logged |
| Text embeddings | all-MiniLM-L6-v2 via **ONNX Runtime** + Hugging Face `tokenizers` | semantic search, content similarity |
| API | **FastAPI**, Pydantic, SQLAlchemy 2, Uvicorn, PyJWT | the recommendation service |
| Database and auth | SQLite (development), **Supabase** Postgres with row-level security + Supabase Auth (production) | users, ratings, watchlists, events |
| Web app | **Next.js 16** (App Router), React 19, TypeScript, Tailwind CSS 4, TanStack Query, Recharts | the user interface |
| Quality | pytest, ruff, ESLint, TypeScript, GitHub Actions, Playwright (browser check) | Chapter 10 |
| Hosting (free tiers) | Vercel (web), Hugging Face Spaces with Docker (API), Supabase | Section 9.8 |

## 9.2 Repository structure

```
movie-recommendation/
├── ml/                    research + training: the "recsys" Python package
│   ├── src/recsys/
│   │   ├── data/          download (checksummed, resumable), parsers, schema, split, TMDB
│   │   ├── eval/          metrics and the shared full-ranking evaluator
│   │   ├── models/        11 models behind one Recommender interface
│   │   ├── tuning/        Optuna search spaces and the equal-budget tuner
│   │   ├── hybrid/        candidates, features, LightGBM ranker, MMR
│   │   ├── content/       viewer keywords, movie texts, ONNX sentence encoder
│   │   ├── serving/       torch-free scorers, bundle export, recommendation engine
│   │   ├── experiments/   configs, protocol runner, MLflow tracking, reports
│   │   └── analysis/      EDA, significance tests, figures
│   ├── configs/           one YAML per experiment (*-tuned.yaml are generated)
│   ├── notebooks/         EDA, Week 1 baselines, full results (sources in notebooks/src)
│   ├── scripts/           train_app_models.sh: the app's MovieLens-32M models, end to end
│   └── tests/             115 tests
├── backend/               FastAPI service (app/), 29 tests, Dockerfile
├── frontend/              Next.js web app
├── supabase/migrations/   production schema with row-level security
├── deploy/                Hugging Face Space builder, smoke test
└── docs/                  plan, report chapters, results tables, figures
```

About 6,400 lines of Python in the ML package, 900 in the API and 2,900 lines of TypeScript in the web app
(excluding tests).

## 9.3 The research pipeline

Everything is driven by one command-line tool, `recsys`, and a YAML config per experiment:

| Command | What it does |
|---|---|
| `recsys download ml-1m` / `prepare` / `eda` | fetch and verify the dataset, convert it to the shared schema, produce statistics and figures |
| `recsys split -c ml-1m` | per-user chronological 80/10/10 split, with leakage checks and a fingerprint |
| `recsys benchmark -c <config>` | train → early-stop on validation → refit on train + validation → test, for every model; logs everything |
| `recsys tune -c ml-1m` | Optuna on validation with an equal budget per model; writes `ml-1m-tuned.yaml` |
| `recsys hybrid -c ml-1m-tuned` | the Red Box hybrid and its MMR variants |
| `recsys rating -c ml-1m` | rating prediction (RMSE/MAE) |
| `recsys content ml-1m` | viewer keywords, movie texts and sentence embeddings |
| `recsys export -c ml-1m-tuned` | the serving bundle for the API |
| `recsys report -c <config>` | result tables (Markdown + CSV) rebuilt from MLflow |

**Reproducibility and cost control.**
- **Run identity.** Every run is identified by a hash of its exact settings plus the split fingerprint.
- **Caching.** Re-running a benchmark skips any model already run with identical settings, and fitted
  models are saved by the same hash. Hours of CPU training are never repeated: the hybrid, the reports and
  the export all reuse saved models.
- **Downloads.** All datasets are verified by SHA-256, and large downloads resume after a broken connection.

**Notable implementation work**
- **LightGCN.** The normalised adjacency matrix is symmetric, so a custom autograd function reuses the
  same sparse product in the backward pass instead of building a transposed copy. This made graph
  training practical on a CPU.
- **SASRec.**
  - Padding-masked attention with right-aligned positions, and length-bucketed batches.
  - Xavier initialisation: with the default initialisation, the model did not learn.
- **Fair early stopping.** Every iterative model early-stops on validation NDCG@10, then is refitted on
  train + validation for the epochs it found best.

## 9.4 The serving bundle

`recsys export` turns trained models into a folder the API can load with NumPy alone:

| File | Content |
|---|---|
| `ease.ease.npy` | EASE item–item matrix (3,458 × 3,458) |
| `ials.ials.npz` | iALS item factors and settings; users are folded in at request time |
| `sasrec_ce.sasrec.npz/.json` | SASRec weights, run by a **NumPy re-implementation** of the Transformer |
| `markov.*.npz` | first-order transition counts |
| `ranker.txt` | the serving LightGBM ranker |
| `items.parquet` | catalogue: titles, years, genres, popularity, keywords, rating statistics, TMDB/IMDb ids |
| `neighbours.npz`, `seeds.npy`, `genres.npz` | similar-film lists, onboarding seeds, genre matrix |
| `text_embeddings.npy`, `text_model/` | film embeddings and the ONNX encoder for queries |
| `item_stats.npz` or `dataset_users.npz` | global item statistics for the ranker's features (see Section 9.8) |
| `manifest.json`, `offline_results.csv` | provenance (split fingerprint, settings hashes), test metrics, results table |

**Two safeguards are tested.**
- **The NumPy SASRec matches PyTorch.** Its outputs equal the PyTorch model's to numerical precision.
- **No train/serve skew.** The ranker's features computed live for one user equal the features computed
  offline for the same user.

The second test caught a real bug during development. Item popularity and item–item similarity were being
computed from the single user's history instead of the whole dataset.

## 9.5 The API (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, models, semantic-search availability, the deployed model's test scores |
| `GET /movies`, `/movies/genres`, `/movies/{id}` | browse, title search, genre filter, details (with keywords and rating statistics) |
| `GET /movies/{id}/similar?mode=cf\|content` | similar films: collaborative or by story/themes |
| `GET /search/semantic?q=` | natural-language search |
| `PUT/DELETE /ratings/{id}`, `GET /me/ratings` | rate (half stars), remove, list |
| `PUT/DELETE /watchlist/{id}`, `GET /me/watchlist` | watchlist |
| `GET /onboarding/seeds`, `POST /onboarding`, `GET /me/profile` | onboarding and profile |
| `GET /recommendations?model=&k=&diversity=` | the personalised list, each item with a reason |
| `GET /recommendations/feed` | home rows: top picks, because you liked X, hidden gems, trending |
| `GET /lab/users`, `GET /lab/compare`, `GET /lab/metrics` | Model Lab and the results dashboard |
| `POST /events` | click logging (impressions are logged automatically) |
| `POST /auth/register`, `/auth/login`, `/auth/password`, `/auth/logout-all`, `GET /auth/me` | accounts (Section 9.9) |
| `PATCH /me/profile`, `GET /me/export`, `POST /me/delete` | profile, download all my data, delete my account |
| `GET /ops/metrics` | operators: live click-through rates and the A/B test (Section 9.10) |

**Design points**
- **Startup and requests.** The bundle is loaded once at startup. Each request computes recommendations
  from the user's current history: no cache, no per-user state.
- **Impression logging.** Impressions are written in a background task after the response is sent.
- **Authentication has three modes.**
  - *local* (default): the API's own email + password accounts (Section 9.9).
  - *supabase:* a Supabase access token, verified with the project's public keys (JWKS) or the legacy
    shared secret.
  - *dev:* an `X-User-Id` header, used only by tests.
- **Logging and headers.** A middleware logs one line per request (method, path, status, duration) and adds
  standard security headers; query strings, bodies and tokens are never logged.
- **Defence in depth.** Row-level security in Postgres means a user can only ever read and write their own
  rows, even if a client reached the database directly.
- **Typed responses.** Every response is a Pydantic model, so `/docs` documents the whole API
  automatically.

## 9.6 The web application

The web app is a single Next.js project. Its pages render as static shells, and the browser fetches data
from the API with TanStack Query.

| Page | What the user can do |
|---|---|
| Landing | read what Red Box does; live numbers from the API |
| Sign in | create an account or sign in with email + password |
| Account | change name or password, sign out everywhere, download my data, delete my account |
| Operations | operators only: live click-through rates per row and the A/B-test result |
| Onboarding | pick genres, rate seed films with half-star ratings, see first recommendations |
| For you (home) | top picks with a diversity slider; "Because you liked…", hidden gems, trending; rate any card in place |
| Movie page | details, viewer keywords (each a search link), average rating, own rating, watchlist, "why you're seeing this", similar films by viewers or by story |
| Search | "Describe it" (semantic) or by title with a genre filter; shareable URLs |
| Library | edit or remove ratings; watchlist |
| Model Lab | choose yourself or a MovieLens viewer; each model's top 10 side by side; hover to find a film in every column; overlap heatmap; MMR slider |
| Results | offline results for both datasets: headline gain, ranking accuracy, accuracy vs coverage, training time, full table, report figures |

**Design and accessibility**
- **Look.** A dark "screening room" theme with one accent colour.
- **Posters without artwork.** Films without TMDB artwork get a typographic poster tinted by genre.
- **Charts.**
  - One series each, with the hybrid highlighted.
  - The colours were checked with a colour-vision-deficiency validator against the dark background.
  - Every chart has tooltips and a table view.
- **Inputs.** Star ratings are keyboard-operable sliders with screen-reader labels.
- **Phones.** Every page was checked at phone width (390 px).

## 9.7 Engineering problems solved

| Problem | Solution |
|---|---|
| MovieLens has 77% same-second ratings, so "chronological" order was ambiguous | seeded random tie-breaking: reproducible and independent of item ids |
| A PyTorch-free server could not run SASRec | NumPy re-implementation, tested against PyTorch |
| Live features differed from training features (train/serve skew) | global item statistics shipped in the bundle; an equality test |
| Large downloads failed midway | resumable HTTP Range downloads with checksums |
| The CPU makes every experiment slow | settings-hash caching of runs and models; sequential background queues |
| "Hidden gems" were noisy for rarely watched films | quality filter plus a content-based taste profile (Section 8.6) |
| MovieLens histories may not be redistributed in a public deployment | ship aggregate item statistics instead of user histories; a test proves recommendations are identical |

## 9.8 Deployment

The deployment is described step by step in `docs/DEPLOYMENT.md`. All three hosts are free tiers.

- **API.** `deploy/huggingface/build_space.py` assembles a Docker Space (API code + model bundle) and can
  publish it, with the database and auth settings stored as Space secrets. The public Space contains:
  - the trained weights;
  - the catalogue;
  - aggregate item statistics;
  - **no raw MovieLens user histories**, which the MovieLens licence does not allow redistributing.

  The Model Lab therefore compares models on the signed-in user's own ratings.
- **Web app.** Deployed to Vercel with one environment variable: the API's address.
- **Database.** Any free hosted Postgres (Supabase or Neon). The API creates its tables at startup, adds new
  columns to existing databases, and switches on row-level security.
- **Check.** `deploy/smoke_test.py` checks a live deployment end to end (Chapter 10).

## 9.9 Accounts and security

Accounts are part of the API, so the system needs no third-party identity service.

| Concern | Implementation |
|---|---|
| Password storage | scrypt (N = 2^15, r = 8, p = 1: 32 MB of memory per hash), a random 16-byte salt per password, constant-time comparison; parameters stored with each hash so the cost can be raised later |
| Sessions | JWT signed with HS256 (key from `REDBOX_JWT_SECRET`), 14-day expiry, carrying the account's *token version*; changing the password or "sign out everywhere" increments the version and so revokes every earlier token |
| Brute force | at most 10 failed logins per email and per client address in 15 minutes (sliding window) |
| Account enumeration | a wrong password and an unknown email return the same message, and an unknown email still runs one hash so both take the same time |
| Data rights | users can download everything stored about them (JSON) and delete their account with all their data (password-confirmed) |
| Database | on Postgres, row-level security is switched on for every table: the API (table owner) is unaffected, while a hosted database's public REST API cannot read anything, password hashes included |

## 9.10 Online evaluation (A/B testing)

Offline metrics (Chapter 11) predict how users will respond. An online test measures it. The API runs
one experiment at a time.

- **Assignment.** A user's group is a stable hash of the experiment name and their id, so it never changes
  between devices or requests, and the groups are balanced in expectation. (A test checks 2,000 users split
  45–55%.)
- **The first experiment.** It compares the *Top picks* row ranked by accuracy alone (`control`) with light
  MMR diversity (`diverse`, 0.2). This is the online counterpart of the offline accuracy–diversity
  trade-off in Section 11.5.
- **Measurement.**
  - Every impression and click is logged with its row and the user's group.
  - Users who move the diversity slider themselves are logged as `manual` and left out of the comparison.
- **Analysis.** The operators' page shows:
  - the click-through rate per row and per group;
  - the difference between the groups with a 95% confidence interval;
  - a two-proportion z-test;
  - activity numbers and catalogue reach (the share of the catalogue that was shown at all).
