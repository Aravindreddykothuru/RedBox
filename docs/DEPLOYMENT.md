# Deploying Red Box (free tiers only)

**Current state:**
- Website: **live** at https://redbox-recommender.vercel.app (Vercel project `redbox-recommender`), built
  to call the API at https://aravindreddy111-redbox-api.hf.space.
- API: **ready to publish** to the Space `aravindreddy111/redbox-api`; the folder is built and passed the
  smoke test locally (`dist/hf-space`). Publishing needs a Hugging Face token with *write* permission
  (the one saved on this machine can only read). Then run, from the repository root:

  ```bash
  REDBOX_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  REDBOX_CORS_ORIGINS='["https://redbox-recommender.vercel.app"]' \
  uv run --project ml python deploy/huggingface/build_space.py \
      --push aravindreddy111/redbox-api --backup-repo aravindreddy111/redbox-data
  ```
- Database: SQLite on the Space, backed up to the private dataset `aravindreddy111/redbox-data` (option B
  below).

```
Browser ──► Next.js on Vercel ──► FastAPI on a Hugging Face Space (Docker) ──► Postgres (Supabase or Neon)
```

Accounts are built into the API: email and password, with hashed passwords and signed tokens. So the only
outside pieces are three free hosts, each needing an account in your own name. Follow the steps in order:
each step needs the address the previous one gives you. (The limits below were current when this was
written; check each provider's pricing page.)

| What | Where | Free tier | Account needed |
|---|---|---|---|
| Website | Vercel | Hobby plan (non-commercial) | Vercel (sign in with GitHub) |
| API + models | Hugging Face Spaces, Docker SDK | CPU basic: 2 vCPU, 16 GB RAM; sleeps after 48 h idle; **no permanent disk** | Hugging Face |
| Database | Supabase *or* Neon (Postgres), *or* a private Hugging Face dataset holding the SQLite file | Supabase: 500 MB, pauses after a week idle · Neon: 0.5 GB | either one, or none beyond Hugging Face |

**Why a hosted database.** The free Space has no permanent disk, so the API's default SQLite file would be
wiped at every restart. Accounts and ratings live in Postgres instead (option A), or the SQLite file is
copied to a private Hugging Face dataset and brought back at every start (option B).

## 0. Check everything locally first

```bash
cd ml && uv run pytest && cd ../backend && uv run pytest && cd ../frontend && npm run lint && npm run build
# API + site running locally (see README), then:
python deploy/smoke_test.py --api http://127.0.0.1:8000 --web http://localhost:3000
cd frontend && npm run e2e         # the browser journey
```

## 1. Database

**Option B, no extra account: SQLite with a backup in a private Hugging Face dataset.** Add
`--backup-repo <hf-username>/redbox-data` to the push in step 2 and skip the rest of this step. The API
then downloads the latest copy before it opens the database and uploads a consistent snapshot every 10
minutes when something changed, and once more when it stops (`backend/app/backup.py`). The dataset is
private; the Space reads and writes it with your token, stored as the Space secret `HF_TOKEN`. A restart
can lose at most the last 10 minutes.

**Option A: Postgres.**

1. Create a free Postgres database:
   - **Supabase:** create a project, then copy *Project Settings → Database → Connection string → Session
     pooler*.
   - **Neon:** create a project and copy its connection string.
2. Change the string's scheme to `postgresql+psycopg://`.

You don't run any SQL yourself: the API creates its tables at startup. On Postgres it also switches on
row-level security, so the hosting service's public data API can't read anything, password hashes
included.

*Using Supabase's own sign-in instead* (`REDBOX_AUTH_MODE=supabase`): run
`supabase/migrations/20261119000000_init.sql` in the SQL editor, and see the README's
"Accounts" section.

## 2. API on a Hugging Face Space

1. Create a **write** token at huggingface.co/settings/tokens.
2. From the repository root, run:

```bash
# PowerShell: $env:NAME="value" ; bash: export NAME=value
export HF_TOKEN=hf_...                                    # never commit it
export REDBOX_DATABASE_URL='postgresql+psycopg://user:password@host:5432/dbname'
export REDBOX_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export REDBOX_ADMIN_EMAILS='["you@example.com"]'      # who may open the live-metrics page
export REDBOX_CORS_ORIGINS='["https://<your-app>.vercel.app"]'   # fill in after step 3, push again

uv run --project ml python deploy/huggingface/build_space.py --push <hf-username>/redbox-api
```

The script does four things:
- assembles `dist/hf-space/` (the API code, the model bundle and a Dockerfile);
- creates the Space;
- stores the database URL and signing key as **Space secrets** and the rest as variables;
- uploads everything.

**Optional: test the exact image first with Docker Desktop.**

```bash
cd dist/hf-space && docker build -t redbox-api:space . && docker run --rm -p 7860:7860 redbox-api:space
python deploy/smoke_test.py --api http://127.0.0.1:7860      # in another terminal
```

The Space then builds for about 5 minutes. When it is ready, the API is at
`https://<hf-username>-redbox-api.hf.space` (try `/health` and `/docs`).

**About the data.** The public Space does *not* contain raw MovieLens viewing histories, because the
MovieLens licence does not allow redistributing them. It ships the trained model weights and aggregate item
statistics instead. Recommendations are identical (there is a test for this). The only difference: the
Model Lab compares models on the signed-in user's own ratings, not on MovieLens viewers. Use
`--include-histories` only for a **private** Space.

## 3. Website on Vercel

```bash
cd frontend
npx vercel link                      # create/link the project (Root Directory: frontend)
npx vercel env add NEXT_PUBLIC_API_URL production          # https://<hf-username>-redbox-api.hf.space
npx vercel --prod
```

Then put the Vercel address into `REDBOX_CORS_ORIGINS` and push the Space again (step 2).

## 4. Smoke test the live system

```bash
python deploy/smoke_test.py --api https://<hf-username>-redbox-api.hf.space --web https://<your-app>.vercel.app
```

The checks cover:
- health, search, film pages and similar films;
- semantic search;
- with a throwaway account it creates and deletes again: onboarding, real-time updates, never
  re-recommending a rated film, and response time;
- the website's pages.

The first request after a long pause wakes the sleeping Space. It can take a minute, so open the site
once before a demo.

## After launch

- **New releases keep coming by themselves.** Once a day the API asks Wikidata for this year's and last
  year's releases in English, Hindi, Telugu, Tamil, Kannada and Malayalam, and adds the ones it doesn't
  have (`backend/app/releases.py`; `REDBOX_RELEASES_REFRESH_HOURS`, default 24). Operators can see the
  status at `GET /ops/releases` and ask for a look now with `POST /ops/releases/refresh`. On the Space they
  are kept until it restarts, and the next daily look adds them again.
- **New films learn from ratings at once** (warm-up), and graduate into every model at the next
  retraining once 20 app users like them (see below).

- **Live metrics:** sign in with an email listed in `REDBOX_ADMIN_EMAILS`; *Operations* appears in the
  menu. It shows click-through rate per row and the A/B test result.
- **Retraining with the app's ratings:**
  1. Bring the ratings in: `REDBOX_TOKEN=<operator token> uv run recsys import-app-ratings <API URL>/ops/ratings.csv`, from `ml/`.
  2. Retrain: `bash ml/scripts/train_app_models.sh --refresh`. Every model retrains because the data
     changed, so allow 4–5 hours on a laptop CPU.
  3. Push the Space again.

## Optional: posters and plot summaries (TMDB)

1. Create a free TMDB account and an API Read Access Token.
2. Fetch the metadata and rebuild the bundle:

```bash
cd ml
export TMDB_READ_TOKEN=...            # never commit it
uv run recsys tmdb ml-32m             # posters, overviews, cast (cached in data/raw/tmdb)
uv run recsys content ml-32m          # re-embed film texts, now with plot summaries
uv run recsys export -c ml-32m        # rebuild the bundle with posters + content
```

3. Push the Space again (step 2).

The site shows real posters automatically once `poster_url` is present.

## Local alternative: Docker

```bash
docker compose up --build             # Postgres (pgvector) + the API on http://localhost:8000
```
