# Chapter 10: Testing

Testing covered five levels:
1. **unit tests** of the mathematics;
2. **integration tests** of the research pipeline;
3. **API tests;**
4. **checks of the web application;**
5. **end-to-end tests** of the running system.

Steps 1–4 run automatically on every push (GitHub Actions). The end-to-end checks were run against the
local stack, and the same smoke test is used against the live deployment.

## 10.1 Strategy

| Level | Tool | What is verified |
|---|---|---|
| Unit (ML) | pytest | metrics against hand-computed values; split leakage; each model's maths against an independent computation |
| Integration (ML) | pytest on small synthetic datasets | the protocol, tuning, hybrid, export and serving engine as a whole |
| API | pytest + FastAPI TestClient, on a tiny generated bundle | every endpoint, authentication, user isolation, real-time behaviour |
| Web app | ESLint, TypeScript (strict), production build | type safety, React rules, every route compiles and prerenders |
| End to end | Playwright driving headless Microsoft Edge; `deploy/smoke_test.py` | the real system: browser → web app → API → models → database |
| Performance | smoke test timings | latency of the main endpoints |

## 10.2 Automated test suites

**ML package: 115 tests**

| File | Tests | Examples of what is checked |
|---|---:|---|
| `test_metrics.py` | 12 | NDCG, recall, precision, MRR, MAP, hit rate, coverage, novelty, ILD and Gini equal hand-computed values |
| `test_evaluator.py` | 5 | full-catalogue ranking excludes training items; per-user results add up |
| `test_split.py` | 10 | chronological parts; disliked ratings dropped; small users train-only; cold items removed from test; same-second ties seeded and order-independent; the leakage check catches leakage; save/load with tamper detection |
| `test_movielens.py`, `test_download.py`, `test_eda.py`, `test_tmdb.py` | 23 | parsing (incl. the MovieLens-32M time window, k-core and genre mapping), schema validation, checksum verification, resumable downloads, statistics, TMDB caching |
| `test_models.py` | 9 | ItemKNN and UserKNN against hand-computed scores; EASE equals per-item ridge regression; BPR's sampler never returns positives; seen items are never recommended |
| `test_deep_models.py` | 18 | iALS half-step equals weighted least squares; LightGCN's adjacency by hand and its custom gradient against autograd; SASRec is causal and padding-invariant; shuffling order destroys its sequence signal; every model learns planted taste groups |
| `test_tuning.py` | 11 | trial 0 equals the default settings and lies inside the search space; config overrides respected; every space's values within range |
| `test_hybrid.py` | 6 | seeded, bounded user sampling for large exports; the candidate union excludes history and labels targets correctly; the ranker learns a planted signal; MMR trades relevance for diversity; hybrid scores round-trip through the evaluator |
| `test_rating.py` | 3 | global mean, biases and biased MF on known data |
| `test_results_analysis.py` | 3 | significance tests (Wilcoxon + Holm), activity and session buckets |
| `test_serving.py` | 7 | NumPy SASRec equals PyTorch; scorers equal the trained models; iALS fold-in is exactly the ALS user step; **live features equal offline features**; the serving path never imports PyTorch; a public bundle without user histories gives identical recommendations |
| `test_content.py` | 8 | keyword extraction and stop tags; movie-text format; encoder mean-pools only real tokens; semantic search and content similarity; bundle alignment by movie id; hidden-gem eligibility; real MiniLM separates meanings (when downloaded) |

**API: 29 tests** (12 for the catalogue and recommendations, 10 for accounts, 6 for online evaluation, 1 for headers)

| Test | Verifies |
|---|---|
| health | status, model list |
| catalogue browse, search, detail | paging, popularity order, title search, genre filter, 404 for unknown films, similar films |
| ratings drive recommendations in real time | after rating films from one taste group, the next response recommends that group, and rated films never reappear |
| rating validation and delete | half-star steps enforced (422), deletion |
| users are isolated | one user never sees another's ratings |
| onboarding flow | seeds respect chosen genres; onboarding stores ratings and returns recommendations |
| feed for new vs active users | a new user gets popular rows only; an active user gets personal rows first |
| Model Lab and metrics | every model's list; 404 for unknown dataset users; metrics endpoint |
| watchlist, events and bad model | watchlist round trip, event logging, 422 for an unknown model |
| Supabase tokens | missing or forged tokens → 401; a valid token works; browsing needs no sign-in |
| semantic search and content similarity | 503 without a text model; correct results with one; validation of short queries; keywords on detail pages |
| Model Lab diversity | the diversity slider changes only the hybrid column |
| Accounts: sign-up and sign-in | case-insensitive emails; duplicates refused (409); weak passwords and malformed emails refused (422) |
| Accounts: no enumeration | a wrong password and an unknown email give identical answers |
| Accounts: tokens | personal endpoints need a valid token; a tampered token and the dev header are refused; users only see their own data |
| Accounts: revocation | a password change and "sign out everywhere" invalidate every earlier token |
| Accounts: brute force | after N failed logins even the right password waits (429) |
| Accounts: data rights | profile update; data export contains ratings, watchlist and events; deletion (password-confirmed) removes everything and frees the email |
| A/B assignment | the same group on every call; balanced (45–55% over 2,000 users); a new experiment reshuffles; switched off = one group |
| A/B statistics | the z-test matches a hand computation (100/1000 vs 130/1000 clicks: p between 0.03 and 0.04) |
| Impression logging | impressions and clicks carry the user's group; top picks are not double-counted; slider users are "manual" |
| Live metrics access | operators only: 401 without a token, 403 for normal accounts |
| Schema migration | a database created by the first release gains the new column and keeps its data |

Result: **all 197 automated tests pass** (131 for the ML package, 66 for the API), together with lint and format checks (ruff, ESLint) and the
TypeScript compiler.

## 10.3 End-to-end test of the web application

A Playwright script (`frontend/e2e/journey.mjs`, run with `npm run e2e`) drove headless Microsoft Edge
through a complete user journey against the production build of the web app and the real model bundle:

| # | Step | Expected | Result |
|---|---|---|---|
| 1 | Open the site in a new tab | the opening animation plays, lifts by itself, and any key skips it | Pass (skipped in about 350 ms) |
| 2 | Landing page | the predicted top ten and the FAQ load | Pass |
| 3 | Create an account (email + password), pick 3 genres | onboarding opens with seeds from those genres | Pass |
| 4 | Rate 12 seed films, finish | home feed with personal rows; every poster in view loads | Pass (39 of 39 posters) |
| 5 | Rate the first top pick 5 stars from the feed | it disappears and the list re-ranks immediately | Pass |
| 6 | Move the diversity slider | top picks change | Pass |
| 7 | Open a recommended film | the reason, its source model and the ranker's TreeSHAP breakdown | Pass |
| 8 | Switch similar films to "story and themes"; add to watchlist | content-based list; watchlist saved | Pass |
| 9 | Semantic search "time travel" | films about time travel, found by story and tags, not title | Pass (*Time Freak*, *Primer*, *11.22.63*, …) |
| 10 | Title search "matrix" | *The Matrix* first | Pass |
| 11 | Model Lab: you, then a MovieLens viewer | six model columns, overlap heatmap | Pass |
| 12 | Results page: Amazon, then MovieLens-32M | charts and tables; the live app's tab with its significance test | Pass (p < 10⁻⁸⁶) |
| 13 | Library: ratings, watchlist | ratings editable; watchlist shows the saved film | Pass |
| 14 | Import a Netflix viewing-history file | 4 films matched; a TV episode and an unknown title left out; imported films count as liked | Pass (*RRR*, *Baahubali 2*, *Dangal*, …) |
| 15 | Watch together with a friend who loves *Inception*, "Nobody unhappy" | one list, each film with both members' fit, ordered by the lower fit | Pass |
| 16 | Sign out; sign in with a wrong password, then the right one | wrong password refused; signed in again | Pass |
| 17 | Account page; Operations page as a normal user | account details shown; operators-only notice | Pass |
| 18 | Home: new releases; filter English, Hindi, Telugu, Tamil, Kannada, Malayalam | each language's own films, ranked from their story, with reasons that name the language | Pass |
| 19 | Switch the language to Telugu and back | navigation, rows, genres and reasons translated | Pass |
| 20 | Repeat home and results at phone width (390 px) | no horizontal overflow | Pass |
| – | Throughout | no browser console errors, failed requests or HTTP errors | Pass |

Screenshots from this run were also reviewed by eye. That review found layout issues that automated checks
cannot see, and they were fixed:
- a row misaligned by scroll snapping;
- crowded mini-posters;
- an uneven poster palette;
- posters still downloading showed as empty boxes: a poster now fades in over the film's typographic cover,
  which stays if the image never loads;
- a film picker's thumbnail made its button read "Poster of Inception Inception 2010" to screen readers:
  the thumbnail is now hidden from them, so the button reads "Inception 2010";
- a new Tamil film showed a rainfall map of India, and others showed an actor's photo: their Wikidata
  links led to articles about something else. Films whose article does not describe them are now left
  out (38 of 2,029; Section 8.5);
- after the films' data was fetched, Wikimedia's image server throttled posters for a while (HTTP 429),
  and a third of a page's posters did not load: posters now come through the API's own cache
  (`GET /posters/{id}.webp`), and only a poster not cached yet is loaded from Wikipedia.

## 10.4 Smoke test and performance

`deploy/smoke_test.py` uses only the Python standard library, so it can check any deployment. Against the
local stack (API + production web build), all **18 of 18** checks passed:

| Check (serving the MovieLens-32M models, 8,487 films) | Measured |
|---|---|
| Title search | 11 ms |
| Similar films | 25 ms |
| Semantic search (after the encoder's one-time load of about 1 s) | 16 ms |
| Onboarding → recommendations change; rated films never recommended | pass |
| `/recommendations` (k = 20), median of 5 | **52 ms** |
| Web pages `/`, `/search`, `/results`, `/lab` | 4–28 ms each |

**The deployment image.** The Hugging Face Space folder was built into a Docker image and the smoke test
was run against a container from it, with the earlier MovieLens-1M bundle; the MovieLens-32M bundle is
larger, so these figures are to be measured again when it is deployed. All 10 API checks passed:

| Measure | Result |
|---|---|
| Image size | 1.05 GB, with no PyTorch |
| Startup time | ready in 10 s |
| Memory | 444 MB, against 16 GB on the free tier |
| `/recommendations`, median | 68 ms inside Docker |

The requirement NFR-5 (200 ms) is met with a wide margin on the development laptop. These timings were
taken while a model was training on the same CPU. They must be re-measured on the free 2-vCPU server
after deployment.

## 10.5 Defects found by testing

| Defect | Found by | Fix |
|---|---|---|
| Ranker features computed from one user's history instead of the dataset (train/serve skew; latency 223 ms) | the live-equals-offline feature test | global item statistics in the bundle (latency 36 ms) |
| A route function shadowed an imported helper (`movie_detail`) | API tests | renamed the route |
| Missing values (`pd.NA`) broke optional catalogue fields | API tests | a single optional-field helper |
| Export crashed: a variable shadowed the ranker's feature table | running the export | renamed |
| Hidden-gems row nearly empty, then poorly matched | browser journey screenshots | new gem definition (Section 8.6) with a test |
| The container crashed at startup: LightGBM needs the OpenMP library (`libgomp1`), missing from the slim base image | running the deployment image | installed in both Dockerfiles |
| The image carried uv's 600 MB download cache | inspecting the image layers | `uv sync --no-cache` (1.64 GB → 1.05 GB) |
