# Red Box API

A FastAPI service that serves personalized movie recommendations from an exported **model bundle**. It
runs without PyTorch: all models run in NumPy, plus a LightGBM ranker and a small ONNX text encoder.

It also handles:
- **real accounts:** email and password;
- **an online A/B test** of the recommendations;
- **live metrics** for operators.

## Run locally

```bash
# 1. The model bundle: the app's MovieLens-32M models (or the research ML-1M ones)
bash ml/scripts/train_app_models.sh                 # -> data/bundles/ml-32m (several hours on a CPU)
# or: cd ml && uv run recsys export -c ml-1m-tuned  # -> data/bundles/ml-1m

# 2. Settings (once): a token-signing key, and your email as an operator
cp backend/.env.example backend/.env                # then fill in REDBOX_JWT_SECRET, ADMIN_EMAILS

# 3. Start the API (SQLite database, real accounts)
cd backend && uv run uvicorn app.main:app --reload  # docs at http://127.0.0.1:8000/docs
```

**Full stack with Postgres:** run `docker compose up --build` from the repository root.

## Accounts (`REDBOX_AUTH_MODE`)

| Mode | Who signs users in | Use it for |
|---|---|---|
| `local` (default) | this API: `/auth/register`, `/auth/login` | real use, with no third party |
| `supabase` | Supabase Auth; the API verifies its tokens | when you prefer Supabase's hosted accounts |
| `dev` | nobody: the `X-User-Id` header names the user | tests and quick experiments only |

How `local` accounts are protected:
- **Passwords:** hashed with scrypt (memory-hard, with a random salt per password), never stored or logged.
- **Tokens:** signed JWTs that expire after 14 days. A password change or "sign out everywhere"
  revokes every earlier token at once.
- **Login attempts:** limited to 10 failures per email and per address in 15 minutes.
- **No account enumeration:** a wrong password and an unknown email get the same answer, in the same time.
- **Your data:** users can download all their data and delete their account.
- **Hosted Postgres:** row-level security is switched on, so a hosted database's public REST API (Supabase)
  can't read the tables.

## Endpoints

| Method & path | What it does |
|---|---|
| `GET /health` | status, dataset, models, semantic search, test score of the deployed model |
| `POST /auth/register` · `POST /auth/login` | create an account / sign in → a token |
| `GET /auth/me` · `POST /auth/password` · `POST /auth/logout-all` | who am I, change password, sign out everywhere |
| `PATCH /me/profile` · `GET /me/export` · `POST /me/delete` | edit name/genres, download all my data, delete my account |
| `GET /movies?q=&genre=&sort=&limit=&offset=` | browse and search the catalog |
| `GET /movies/genres` | all genres |
| `GET /posters/{movie_id}.webp` | a film's poster from the API's cache, or a redirect to Wikipedia's file while it is cached; public |
| `GET /movies/trending?k=` | what the hybrid predicts many viewers will watch next (the crowd forecast in `forecast.npz`, built by `recsys forecast`); public |
| `GET /movies/{id}` | details, keywords, rating statistics, plus your rating and watchlist status |
| `GET /movies/{id}/similar?mode=cf\|content` | liked by the same viewers / similar story and themes |
| `GET /search/semantic?q=` | natural-language search |
| `PUT /ratings/{id}` · `DELETE /ratings/{id}` · `GET /me/ratings` | rate (0.5–5 stars, half steps), un-rate, list |
| `PUT/DELETE /watchlist/{id}` · `GET /me/watchlist` | watchlist |
| `POST /me/import/preview` · `POST /me/import` | watch-history import from Netflix, IMDb or Letterboxd: match the uploaded export (never stored), then save the films the user keeps (`app/history_import.py`) |
| `GET /me/watched` · `DELETE /me/watched[/{id}]` | the imported history: films counted as liked unless rated lower |
| `GET /onboarding/seeds?n=&genre=` · `POST /onboarding` · `GET /me/profile` | onboarding |
| `GET /recommendations?k=&model=&diversity=` | personal list with a reason per film; without `diversity`, the user's A/B group decides |
| `GET /recommendations/feed` | home rows: top picks, because you liked X, hidden gems, trending |
| `GET /recommendations/why/{id}` | "Why this pick?": the ranker's score for one film split by signal (TreeSHAP from LightGBM), its rank among your candidates and the models that proposed it |
| `GET /recommendations/new-films?k=&language=` | new films nobody has rated (Telugu, Tamil, Kannada and Malayalam releases), ranked by their story: text mapped to iALS factors (`recsys newfilms`); all languages, or one; `languages` lists them |
| `POST /group` | watch together: you and up to five friends (a few films each), combined by average, least misery or most pleasure |
| `GET /lab/users` · `GET /lab/compare` · `GET /lab/metrics` | Model Lab and offline results |
| `POST /events` | click logging (impressions are logged automatically) |
| `GET /ops/metrics?days=` | operators: click-through rate per row and per A/B group (with a z-test), activity, catalogue reach |

A new rating changes the **very next** response. Recommendations are computed from the current history on
every request: there is no retraining and no per-user cache. Typical latency is tens of milliseconds.

## Online A/B test

Each user is assigned to a group by a stable hash of the experiment name and their id, so they stay in it
on every device. The default experiment compares the "Top picks" row with no diversity re-ranking
(`control`) against light MMR diversity (`diverse`, 0.2).
- **Logging:** impressions and clicks carry the group.
- **Results:** `/ops/metrics` reports each group's click-through rate, the difference with a 95% interval,
  and a two-proportion z-test.
- **Opting out:** users who move the diversity slider themselves are counted as `manual` and left out of
  the test.

## Configuration (environment variables, or `backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `REDBOX_AUTH_MODE` | `local` | `local`, `supabase` or `dev` |
| `REDBOX_JWT_SECRET` | random per start | **set it in production**: signs login tokens |
| `REDBOX_ADMIN_EMAILS` | `[]` | accounts that may read `/ops/metrics` |
| `REDBOX_DATABASE_URL` | SQLite file in `data/` | e.g. `postgresql+psycopg://…` |
| `REDBOX_BUNDLE_DIR` | `data/bundles/ml-32m` if exported, else `ml-1m` | the model bundle |
| `REDBOX_AB_EXPERIMENT` · `REDBOX_AB_VARIANTS` | `topPicksDiversity-1` · `{"control": 0.0, "diverse": 0.2}` | the A/B test (empty name = off) |
| `REDBOX_SUPABASE_URL` · `REDBOX_SUPABASE_JWT_SECRET` | – | `supabase` mode only |
| `REDBOX_CORS_ORIGINS` | `["http://localhost:3000", …]` | the web app's address |
| `REDBOX_POSTER_DIR` | `data/posters` | the poster cache (see below) |

## Posters

Posters come from Wikipedia, whose image server allows an automated client only about one image every
one to three seconds. `GET /posters/{movie_id}.webp` serves a poster from the API's disk cache (342 px
WebP, about 20 KB, cached by browsers for a year). A poster not cached yet is answered with a redirect
to Wikipedia's file, which the browser loads under its own limits, and joins a background queue that
caches it at the pace Wikimedia accepts (`app/posters.py`). To fill the whole cache ahead of time
(about 10,000 posters, several hours; stop and rerun at any time, it resumes):

```bash
uv run python -m app.posters            # most-watched films and new films first
```

## Tests

```bash
uv run pytest      # 29 API tests on a tiny generated bundle, in-memory SQLite
```
