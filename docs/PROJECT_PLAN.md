# Project Plan: Collaborative Filtering for Personalized Movie/Product Recommendations using ML/DL Techniques

| | |
|---|---|
| **Type** | Research comparison + deployed full-stack system |
| **Duration** | 6 weeks, with a 4-week cut line marked |
| **Budget** | Free tools and free tiers only |
| **Hardware** | Laptop CPU (no NVIDIA GPU) + free Kaggle/Colab GPUs for heavy deep-learning runs |
| **Working app name** | Red Box (placeholder, change anytime) |
| **Status** | Weeks 0–6 built and tested: 11 models on ML-1M and Amazon; Red Box hybrid best on both (NDCG@10 0.1790 / 0.0919); order ablation done; semantic search; Next.js web app with all 8 pages; deployment kit ([DEPLOYMENT.md](DEPLOYMENT.md)); report chapters 1–13 in [report/](report/). Waiting on the owner: free accounts to go live (Vercel, Hugging Face, optional Supabase/TMDB), demo video. 4-week cut line applied to ML-32M app models (the app serves ML-1M) |

---

## 1. Summary

We build and fairly compare **11 collaborative-filtering algorithms**, from classic user/item neighborhood methods to graph neural networks and transformers, on **MovieLens-1M** (movies) and **Amazon Reviews 2023** (products).

We then combine the best ones into our **proposed hybrid two-stage recommender**: several models generate candidates, a LightGBM learning-to-rank model orders them, and a diversity step finalizes the list.

Finally we ship it as a **live web app**. Users sign up, rate a few movies, and get personalized, explained recommendations that update the moment they rate something, with no retraining needed. A **Model Lab** page shows every algorithm's recommendations side by side, which doubles as the viva demo.

---

## 2. Problem statement, objectives, research questions

**Problem.** Users face too many choices. Collaborative filtering (CF) learns preferences from user–item interactions alone, but it struggles with:
- **Sparsity:** MovieLens-1M is about 95.5% empty, since each user rates a tiny fraction of the movies.
- **Cold start:** new users and new items have no history.
- **Popularity bias:** models keep recommending the same blockbusters.
- **An open research question:** recent studies show that many deep models do not beat well-tuned classic methods (Ferrari Dacrema et al. 2019; Rendle et al. 2020).

**Objectives**
| # | Objective |
|---|---|
| O1 | Build a reproducible data and evaluation pipeline (chronological splits, full-ranking metrics, no data leakage) |
| O2 | Implement and tune 11 CF algorithms across 3 families: neighborhood, latent factor, deep learning |
| O3 | Compare them on accuracy and beyond-accuracy metrics (coverage, novelty, diversity, popularity bias), with statistical significance tests |
| O4 | Propose a hybrid two-stage recommender and measure whether it beats the best single model |
| O5 | Handle cold-start users (onboarding plus instant fold-in) and cold-start items (content embeddings) |
| O6 | Deploy a full-stack web app with real-time personalization and explanations |
| O7 | Test whether the findings carry over to product recommendation (Amazon) |

**Research questions** (these become the "Results & Discussion" chapter)
- **RQ1:** Do deep models (NeuMF, Mult-VAE, LightGCN, SASRec) beat well-tuned classic CF (ItemKNN, iALS, EASE)?
- **RQ2:** Does the hybrid two-stage ranker improve over the best single model, and which features matter most?
- **RQ3:** How does each model trade accuracy against diversity, novelty and coverage? How do models perform for low-activity (cold-ish) users?
- **RQ4:** Do the conclusions hold on product data (Amazon) as well as movies?

---

## 3. Datasets

| Dataset | Role | Approx. size | Why this one |
|---|---|---|---|
| **MovieLens-1M** | Main research benchmark | 1,000,209 ratings · 6,040 users · ~3,700 movies · user age/gender/occupation | Used in the NCF, LightGCN and SASRec papers, so our numbers are comparable with the literature |
| **Amazon Reviews 2023**, one category (e.g. Video Games), 5-core | Product benchmark (RQ4) | Depends on category | Covers the "/Product" part of the title; different sparsity and behavior from movies |
| **MovieLens-32M**, filtered (≈10k most-rated movies, sampled users) | Trains the models behind the live app | Filtered to fit CPU and memory | Includes recent movies (up to 2023) and TMDB ids, so the app shows posters |
| **TMDB API** (free key) | Posters, overviews, cast, keywords | ~10k movies, fetched once and cached | Movie detail pages and content embeddings for cold-start items |

**Data decisions**
- **Positive feedback:** a rating ≥ 4 counts as a "liked" interaction for top-K recommendation. *Ablation:* treat every rating as positive.
- **Explicit ratings** are kept for a separate rating-prediction experiment (RMSE/MAE).
- **Split:** per-user chronological 80/10/10. Each user's oldest 80% of interactions are used for training, the next 10% for validation (tuning, early stopping, training the ranker) and the newest 10% for test. Every model uses the exact same split, and the model never sees the future.
  - *Confirmed by the EDA (Week 0):* a single global cut-off date would leave only 29.5% of users with any test data.
  - 77% of ratings share the exact same second with another rating by the same user, so same-second ties are broken by a *seeded random key*: reproducible, and not tied to item ids or file order. This matters: for 58% of users a split boundary falls inside such a tie (measured in Week 1).
  - Users with fewer than 3 liked movies are kept in training only and not evaluated: 3 users, plus 2 users with no liked movies at all who are dropped.
  - See [the EDA notebook](../ml/notebooks/01_eda_movielens_1m.ipynb).
- **Licensing:** MovieLens and the Amazon data are for research use, so we never commit the raw data to git and download it with scripts instead. TMDB requires an attribution line in the app footer.

---

## 4. Algorithms

All models are implemented by us in PyTorch/NumPy/SciPy behind one common interface (`fit`, `score`, `recommend`). We are **not** using Surprise or RecBole. Both have install problems on Windows with Python 3.12, and our own implementations are much easier to explain in the viva.

| # | Model | Family | Paper | Core idea (one line for the viva) | Priority |
|---|---|---|---|---|---|
| 1 | MostPop | Baseline | – | Recommend the most-liked items to everyone (sanity floor) | Must |
| 2 | UserKNN | Neighborhood | Resnick 1994 / textbook | Users similar to you liked these | Must |
| 3 | ItemKNN | Neighborhood | Sarwar 2001 | Items similar to what you liked | Must |
| 4 | MF-SVD (explicit) | Latent factor | Koren 2009 | Rating ≈ user vector · item vector + biases (used for the RMSE task) | Should |
| 5 | BPR-MF | Latent factor | Rendle 2009 | Learn to rank liked items above unliked ones (pairwise loss) | Must |
| 6 | iALS | Latent factor | Hu, Koren, Volinsky 2008 | Weighted alternating least squares for implicit feedback | Should |
| 7 | EASE^R | Linear autoencoder | Steck 2019 | Closed-form item-item weight matrix; famously strong and simple | Must |
| 8 | NeuMF | Deep learning | He 2017 | Generalized MF combined with an MLP for non-linear interactions | Must |
| 9 | Mult-VAE | Deep learning | Liang 2018 | Variational autoencoder that reconstructs a user's whole interaction vector | Should |
| 10 | LightGCN | Graph neural network | He 2020 | Propagate embeddings over the user–item graph (neighbors of neighbors) | Must |
| 11 | SASRec | Transformer (sequential) | Kang & McAuley 2018 | Self-attention over the user's watch order to predict the next item | Must |
| ★ | **Red Box Hybrid (proposed)** | Two-stage | Ours, inspired by Covington 2016 | Multi-model candidates → LightGBM LambdaRank → MMR diversity | Must |

### 4.1 Proposed hybrid system (the "Proposed System" chapter)

```
User history ─┬─► LightGCN top-100 ─┐
              ├─► EASE     top-100 ─┼─► union ≈ 200–300 candidates
              ├─► SASRec   top-100 ─┤
              └─► Popular / content fallback ┘
                                   │
                                   ▼
            Stage 2: LightGBM LambdaRank (optimizes NDCG directly)
            features: each model's score + rank, item popularity & age,
            genre match with user's taste, content-embedding similarity to
            user's history, user activity level, demographics (ML-1M)
                                   │
                                   ▼
            Stage 3: MMR re-ranking (accuracy ↔ diversity slider λ),
            remove already-seen items → final top-K + explanation
```

- **No leakage:** the candidate models train on the *train* split. The ranker learns from *validation* labels (the future relative to the candidate models). For the final test, the candidate models are retrained on train + validation.
- **Explanations:**
  - "Because you liked *X*": the item in the user's history whose embedding is closest to the recommended item.
  - "Matches your love of Sci-Fi": the top LightGBM feature contributions (built-in SHAP values via `pred_contrib`).

### 4.2 Cold start
- **New users:** onboarding asks for favorite genres, then for ratings of 10–15 seed movies. Seeds are chosen to be popular *and* divisive, because those give the most information.
- **Instant scoring:** EASE, SASRec and Mult-VAE score a brand-new user directly from their interaction vector, with no retraining. MF and LightGCN use *fold-in*, which estimates the user vector from the items they rated.
- **New items:** a sentence-transformer embeds the TMDB overview, genres and keywords. A small regressor maps this content embedding into LightGCN's item space, trained on movies that already have ratings. A movie with zero ratings can therefore still be recommended. *(Should)*

---

## 5. Evaluation protocol

- **Ranking metrics (main):** Recall@K, NDCG@K, HitRate@K, Precision@K, MAP@K, MRR, for K ∈ {5, 10, 20}.
  - Each user is ranked against the **full catalog**, not a sample of 100 items, because sampled metrics give misleading results (Krichene & Rendle 2020).
  - Items already seen in training are excluded from the ranking.
- **Beyond-accuracy:**
  - catalog coverage@K
  - novelty (mean self-information)
  - intra-list diversity (genre or embedding distance)
  - average popularity and Gini index of recommended items
- **Rating prediction (side experiment):** RMSE and MAE for MF-SVD against the global-mean and bias baselines.
- **Fair comparison:** every model gets the same split and the same tuning budget (30 Optuna trials, optimizing validation NDCG@10).
- **Reliability:** deep models are run with 5 seeds and reported as mean ± std. Significance uses a paired Wilcoxon test on per-user NDCG@10 with Holm correction.

### Experiment list (each becomes a table or figure in the report)
| ID | Experiment | Output |
|---|---|---|
| E1 | All models on ML-1M, all metrics | Main results table |
| E2 | The same on Amazon | Second results table (RQ4) |
| E3 | Significance tests against the best baseline | p-value table |
| E4 | Accuracy vs. novelty/coverage | Scatter plot (RQ3) |
| E5 | Performance by user activity bucket (≤20, 21–50, 51–200, >200 interactions) | Grouped bar chart (cold start) |
| E6 | Ablations: LightGCN layers 1–4, embedding size 32–256, ranker feature groups, MMR λ, rating threshold | Line charts and tables |
| E7 | Rating prediction RMSE/MAE | Small table |
| E8 | Efficiency: training time, ms per user at inference, model size | Table |
| E9 | Hyperparameter importance (Optuna) | Plot |

---

## 6. System architecture

```
┌──────────────────── OFFLINE (laptop CPU / Kaggle GPU) ────────────────────┐
│ download → clean → split → train models → tune (Optuna) → evaluate       │
│                         all runs logged in MLflow                        │
│                                   │                                       │
│ export artifacts: item embeddings (.npy), EASE matrix, SASRec (.onnx),    │
│ LightGBM ranker (.txt), movie metadata, metrics.json                      │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
Browser ─► Next.js on Vercel ─► FastAPI on Hugging Face Spaces ─► Supabase
           (UI, Supabase Auth)   (models loaded in memory,        (Postgres + pgvector:
                                  NumPy + ONNX Runtime +           users, ratings, events,
                                  LightGBM, no PyTorch)            movies, embeddings)
```

**One recommendation request** (`GET /recommendations`):
1. Load the user's ratings from Postgres, served from an in-memory cache when possible.
2. Generate candidates from LightGCN, EASE and SASRec (matrix math in memory).
3. Build features and score them with LightGBM.
4. Apply MMR diversity, filter seen items, attach explanations.
5. Log impressions to the `events` table for later analysis.

Target: under 200 ms per request on the free 2-vCPU server (to be measured, not assumed).

**Real-time personalization:** when the user rates a movie, the next request already uses the new rating. The models don't need retraining, because EASE, SASRec and fold-in all work directly from the user's current history.

---

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Python env | Python 3.12 + **uv** | Fast, reproducible installs with a lock file |
| Data | pandas, SciPy sparse matrices, polars for ML-32M | Sparse matrices are the natural format for CF |
| ML / DL | **PyTorch** (CPU locally, Kaggle/Colab GPU when needed) | Industry standard |
| Ranking | **LightGBM** (LambdaRank) | Fast learning-to-rank on CPU |
| Tuning | **Optuna** | Efficient hyperparameter search with pruning |
| Experiment tracking | **MLflow** (local) | Every run's parameters and metrics recorded, free |
| Content embeddings | sentence-transformers `all-MiniLM-L6-v2` | Small (384-dim), fast on CPU, free |
| Serving | **FastAPI** + NumPy + **ONNX Runtime** + LightGBM | No PyTorch in the server image, so it stays small and starts fast |
| Database + auth | **Supabase** (Postgres + pgvector + Auth), free tier | One free service for DB, vector search and login; you've used it before |
| Frontend | **Next.js** (App Router) + TypeScript + Tailwind + shadcn/ui + TanStack Query + Recharts | Modern, fast, good-looking with little effort |
| Hosting | Vercel (frontend) · **Hugging Face Spaces, Docker** (backend: free 2 vCPU / 16 GB RAM) · Supabase | All free; HF Spaces has enough RAM for the model matrices |
| Quality | pytest, ruff, GitHub Actions CI, Docker Compose for local dev | Professional workflow, also counts toward the report's Testing chapter |

---

## 8. Database schema (Supabase Postgres)

| Table | Key columns |
|---|---|
| `profiles` | `id` (= auth user id), display_name, favorite_genres[], onboarded_at |
| `movies` | `id`, ml_id, tmdb_id, title, year, genres[], overview, poster_path, popularity, `embedding vector(384)` |
| `ratings` | user_id, movie_id, rating (0.5–5), created_at · unique(user_id, movie_id) |
| `watchlist` | user_id, movie_id, added_at |
| `events` | user_id, movie_id, type (impression / click / rate / watchlist), source_model, position, created_at |
| `model_registry` | name, version, dataset, metrics (jsonb), artifact_path, is_active, created_at |

Row-level security is on, so users can only read and write their own ratings, watchlist and events.

---

## 9. API design (FastAPI)

| Method & path | Purpose |
|---|---|
| `GET /health` | Health check, plus the versions of the loaded models |
| `GET /movies/{id}` · `GET /movies?q=&genre=&page=` | Movie details, keyword search and browsing |
| `GET /movies/{id}/similar?mode=cf\|content` | Similar movies by CF embeddings or by content |
| `GET /search/semantic?q=` | Natural-language search ("mind-bending space movies") through pgvector |
| `GET /onboarding/seeds` · `POST /onboarding` | Seed movies to rate; save genres and first ratings |
| `POST /ratings` · `DELETE /ratings/{movie_id}` · `GET /me/ratings` | Rate a movie, remove a rating, list my ratings |
| `GET /recommendations?model=hybrid&k=20&diversity=0.3` | Personalized list with scores and explanations |
| `GET /recommendations/feed` | Home-page rows: Top Picks, Because you liked X, Trending, Hidden Gems |
| `GET /lab/compare?models=...` | Model Lab: the same user's recommendations from each model, plus overlap |
| `GET /lab/metrics` | Offline experiment results for the dashboard |
| `POST /events` | Log impressions and clicks |

The backend verifies Supabase JWTs. Every response is typed with Pydantic, and the OpenAPI docs are generated automatically at `/docs`.

---

## 10. Frontend pages

1. **Landing:** what the app does, plus sign up / log in.
2. **Onboarding:** pick genres, then rate 10–15 seed movies on quick cards. The first recommendations appear immediately afterwards.
3. **Home feed:** rows for *Top Picks for You* (hybrid), *Because you liked X*, *Trending*, *Hidden Gems* (high novelty) and per-genre rows.
4. **Movie detail:** poster, overview and cast, a rating widget, *Why this was recommended*, and similar movies with CF / content tabs.
5. **Search:** keyword search plus semantic natural-language search.
6. **My ratings & watchlist.**
7. **Model Lab** (viva showpiece):
   - choose yourself or any MovieLens user
   - see each algorithm's top-10 side by side
   - drag the diversity slider and watch the list change
   - view the overlap heatmap between models
8. **Results dashboard:** charts from the offline experiments, including NDCG per model, accuracy vs. diversity, cold-start buckets and training time.

---

## 11. Repository structure

```
movie-recommendation/            # in C:\Users\aravind reddy\projects (outside OneDrive)
├── ml/                          # research + training (Python package "recsys")
│   ├── src/recsys/              # (src layout)
│   │   ├── analysis/            # EDA statistics + report figures
│   │   ├── data/                # download, loaders (movielens, amazon), cleaning, splits
│   │   ├── models/              # pop, knn, mf_svd, bpr, ials, ease, neumf, multvae, lightgcn, sasrec
│   │   ├── hybrid/              # candidate generation, feature builder, LightGBM ranker, MMR
│   │   ├── content/             # TMDB fetch, text embeddings, cold-start item mapper
│   │   ├── eval/                # metrics, evaluator, significance tests, reports
│   │   ├── tuning/              # Optuna search spaces
│   │   └── export/              # artifact + ONNX export for serving
│   ├── configs/                 # YAML per dataset/model
│   ├── scripts/                 # CLI entry points: download, train, evaluate, export
│   ├── notebooks/               # EDA, results analysis, figures for the report
│   └── tests/
├── backend/                     # FastAPI service
│   ├── app/ (api/, core/, services/, schemas/, db/)
│   ├── Dockerfile
│   └── tests/
├── frontend/                    # Next.js app
├── supabase/migrations/         # SQL schema + RLS policies
├── docs/                        # this plan, report chapters, diagrams, PPT
├── data/  models/  mlruns/      # git-ignored (downloaded data, artifacts, MLflow runs)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 12. Timeline (6 weeks)

Dates assume we start this week. Write the matching report chapter **every week** rather than all at the end.

| Week | Dates | Focus | Done when… | Report chapter |
|---|---|---|---|---|
| **0** | Nov 19–22 | Setup and research | Repo, env and CI working · ML-1M downloaded · EDA notebook with sparsity/long-tail/activity plots · 20-paper literature list | Abstract, Introduction, Literature Survey |
| **1** | Nov 23–29 | Data pipeline + evaluation harness + classic models | Chronological split with leakage tests · metrics unit-tested against hand-computed examples · MostPop, UserKNN, ItemKNN, BPR-MF, EASE evaluated and logged in MLflow | Requirements (SRS), Methodology part 1 |
| **2** | Nov 30–Dec 6 | Deep-learning models | NeuMF, LightGCN, SASRec (+ Mult-VAE, iALS, MF-SVD) trained and evaluated, using Kaggle GPU where needed | Methodology part 2 (math for each model) |
| **3** | Dec 7–13 | Hybrid + experiments | Optuna tuning for all models · multi-seed runs · hybrid ranker + MMR · Amazon results · experiments E1–E9 done, figures exported | Proposed System, Results & Discussion (draft) |
| **4** | Dec 14–20 | Backend + data for app | ML-32M app models trained and exported · TMDB metadata + embeddings in Supabase · FastAPI endpoints with tests · Docker image runs locally | System Design (architecture, ER, DFD, sequence, class diagrams) |
| **5** | Dec 21–27 | Frontend | All pages working end to end: onboarding → feed → rate → instant update · Model Lab + Results dashboard | Implementation |
| **6** | Dec 28–Jan 3 | Ship + polish | Deployed on Vercel + HF Spaces + Supabase · smoke tests pass · README · demo video · PPT · report final | Testing, Conclusion & Future Scope, References |

**4-week cut line** (if you need to finish early):
- Drop Amazon, Mult-VAE, iALS, MF-SVD, semantic search and new-item cold start.
- Serve the ML-1M models in the app instead of training separate ML-32M ones.
- Merge weeks 4 and 5 into one with a leaner UI.

**Mapping to typical college reviews:**

| Review | Timing | What to show |
|---|---|---|
| Review 1 | after Week 0 | Problem statement, literature survey, objectives, architecture |
| Review 2 | after Week 2 | Working pipeline and first results table |
| Review 3 | after Week 4 | Hybrid results and API |
| Final | after Week 6 | Live app, full report and demo |

Adjust these to your college's actual dates.

---

## 13. Testing strategy

- **ML unit tests:**
  - every metric checked against hand-computed examples (the most important tests in the project)
  - the split never puts a user's future interactions in train
  - models never recommend already-seen items
  - output shapes and ranges are sane
- **Reproducibility:** fixed seeds, configs saved with every MLflow run, and data checksums verified after download.
- **Backend:** API tests with FastAPI's TestClient, auth checks, and a test that a new rating changes the next recommendations.
- **Frontend:** a Playwright smoke test of sign-up → onboarding → feed → rate *(Could)*.
- **Performance:** measure p50/p95 latency of `/recommendations` with a small load script.
- The report's Testing chapter gets a test-case table built from these.

---

## 14. Report deliverables

**Chapters:**
1. Abstract
2. Introduction
3. Literature Survey (with a comparison table of the surveyed methods)
4. Problem Statement & Objectives
5. Software Requirements Specification
6. System Design
7. Methodology (algorithms with equations)
8. Proposed System
9. Implementation
10. Testing
11. Results & Discussion
12. Conclusion & Future Scope
13. References (IEEE format)

**Diagrams:**
- System architecture
- Use-case diagram
- DFD level 0 and level 1
- ER diagram
- Class diagram of the `recsys` package
- Sequence diagram of a recommendation request
- Activity diagram of onboarding
- Hybrid pipeline diagram

**Also:**
- PPT for each review
- 3–5 minute demo video
- GitHub README with results table and screenshots
- Optional: a 6-page IEEE-format paper from the Results chapter

**Core references (all real, all to be cited):**
1. Sarwar et al., *Item-based collaborative filtering recommendation algorithms*, WWW 2001
2. Koren, Bell, Volinsky, *Matrix factorization techniques for recommender systems*, IEEE Computer 2009
3. Hu, Koren, Volinsky, *Collaborative filtering for implicit feedback datasets*, ICDM 2008
4. Rendle et al., *BPR: Bayesian personalized ranking from implicit feedback*, UAI 2009
5. Sedhain et al., *AutoRec: Autoencoders meet collaborative filtering*, WWW 2015
6. Covington, Adams, Sargin, *Deep neural networks for YouTube recommendations*, RecSys 2016
7. He et al., *Neural collaborative filtering*, WWW 2017
8. Liang et al., *Variational autoencoders for collaborative filtering*, WWW 2018
9. Kang & McAuley, *Self-attentive sequential recommendation*, ICDM 2018
10. Steck, *Embarrassingly shallow autoencoders for sparse data*, WWW 2019
11. Wang et al., *Neural graph collaborative filtering*, SIGIR 2019
12. Sun et al., *BERT4Rec*, CIKM 2019
13. Ferrari Dacrema, Cremonesi, Jannach, *Are we really making much progress?*, RecSys 2019
14. He et al., *LightGCN*, SIGIR 2020
15. Rendle et al., *Neural collaborative filtering vs. matrix factorization revisited*, RecSys 2020
16. Krichene & Rendle, *On sampled metrics for item recommendation*, KDD 2020
17. Harper & Konstan, *The MovieLens datasets: History and context*, ACM TiiS 2015
18. Hou et al., *Bridging language and items for retrieval and recommendation* (Amazon Reviews 2023), 2024
19. Ke et al., *LightGBM*, NeurIPS 2017; Burges, *From RankNet to LambdaRank to LambdaMART*, MSR 2010
20. Carbonell & Goldstein, *The use of MMR, diversity-based reranking…*, SIGIR 1998

---

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No local GPU | Slow deep-learning training | ML-1M models train fine on CPU; use Kaggle's free weekly GPU hours for LightGCN/SASRec on ML-32M |
| Scope creep (11 models + app in 6 weeks) | Late submission | Must/Should/Could labels; 4-week cut line already defined; Must items first, always |
| Evaluation mistakes (leakage, sampled metrics) | Wrong conclusions, viva questions | Chronological split, full ranking, unit-tested metrics, one shared evaluator for all models |
| Free hosting sleeps (HF Spaces after inactivity, Supabase pauses after ~1 week idle) | Slow or failed demo | Open the app 10 minutes before every review; keep a recorded demo video as backup |
| EASE matrix memory (10k × 10k floats ≈ 400 MB) | Server RAM | float32/float16 storage; HF Spaces has 16 GB |
| TMDB rate limits | Slow metadata fetch | One-time cached fetch script with retries; metadata stored in Supabase |
| Windows + Python package build issues | Setup delays | uv + wheels-only packages; Docker for the backend |
| Project folder was inside OneDrive and inside a git repo rooted at your whole user profile | Multi-GB cloud sync; accidental commits of personal files | **Resolved in Week 0:** moved to `C:\Users\aravind reddy\projects\movie-recommendation` with its own `git init` |

---

## 16. Key decisions and why (for the viva)

- **Two-dataset benchmark plus a separate app dataset:**
  - ML-1M makes our results comparable with published papers.
  - Amazon tests generalization to products.
  - ML-32M gives the app recent movies with posters.
- **Our own implementations, not libraries:** we can explain every line, and we avoid Windows/Python 3.12 install issues.
- **Full-ranking evaluation with a chronological split:** this is the current best practice, and sampled metrics and random splits overestimate quality.
- **Classic baselines tuned as hard as the deep models:** this answers RQ1 honestly, which is a real open question in the field.
- **Two-stage hybrid:** this is how industry recommenders work (retrieval, then ranking), and it lets us combine graph, linear and sequential signals.
- **No PyTorch in the server:** NumPy + ONNX + LightGBM keeps the Docker image small, cold starts fast, and it fits free hosting.
- **Instant updates without retraining:** the inductive models (EASE, SASRec, fold-in) make the app react to every new rating.

---

## 17. Stretch goals (only after all Must items are done)
- BERT4Rec as a 12th model
- LLM-written explanations using local Ollama (dev only, since free hosting can't run it)
- Interleaving or A/B comparison of two models using the logged `events`
- Retrain the app models from accumulated app ratings with a scheduled GitHub Action
- Admin analytics page (CTR by model, coverage over time)
- Publish an IEEE conference paper

---

## 18. Week 0: first steps checklist

**You (accounts, all free):**
- [ ] TMDB account + API key (themoviedb.org → Settings → API)
- [ ] Supabase project (new, free tier)
- [ ] Kaggle account with phone verification (needed for free GPU)
- [ ] Hugging Face account (for Spaces hosting)
- [ ] GitHub repo for this project (private until submission if you prefer)
- [ ] Ask your guide for the report template, review dates, and whether they want IEEE format

**Me (once you say go):**
- [x] `git init` inside this folder, `.gitignore`, repo skeleton from section 11
- [x] uv Python environment + core dependencies, ruff + pytest + CI
- [x] Dataset download scripts with checksums (ML-1M first)
- [x] EDA notebook: sparsity, rating distribution, long-tail popularity, user activity, time patterns
- [x] Literature survey table draft for Review 1
