# Chapter 5: Software Requirements Specification

> Updated Week 6. Structure loosely follows IEEE 830.
> **Status column:** ✅ implemented and tested · ◐ API implemented and tested, user interface still to come ·
> 🔜 planned (week in brackets).

---

## 5.1 Introduction

**Purpose.** This chapter specifies the requirements of *Red Box*, a system that:
1. compares collaborative-filtering algorithms for top-K recommendation under one rigorous protocol, and
2. serves the best approach to end users through a web application.

**Scope.** The system has two subsystems:
- **Research pipeline:** data acquisition, preprocessing, model training, evaluation and experiment tracking. It runs offline on a laptop CPU and on free cloud GPUs.
- **Web application:** user accounts, onboarding, personalized and explained recommendations that update in real time, search, and a Model Lab that compares algorithms side by side.

**Definitions**

| Term | Meaning |
|---|---|
| CF | collaborative filtering |
| Top-K | the K highest-scored items shown to a user |
| Positive interaction | a rating ≥ 4 |
| Cold start | a user or item with little or no interaction history |
| Full ranking | scoring the whole catalog during evaluation, with no item sampling |

---

## 5.2 Overall description

**Users of the system**

| User class | Needs |
|---|---|
| Researcher (student, guide, examiner) | reproduce experiments, compare models, inspect metrics |
| End user (app visitor) | discover movies they will like, understand why they are recommended |
| Administrator | load data, deploy models, monitor the service |

**Operating environment**
- **Development:** Windows 11 laptop, Python 3.12, CPU only. Kaggle/Colab free GPUs for heavy deep-learning runs.
- **Deployment:** Vercel (frontend), Hugging Face Spaces in a Docker container (backend), Supabase (PostgreSQL, authentication, pgvector). All on free tiers.

**Constraints**
1. Zero budget: free tools and free hosting tiers only.
2. 6-week timeline.
3. Datasets may not be redistributed (MovieLens licence), so data is downloaded by scripts and never committed.
4. TMDB metadata requires attribution.

**Assumptions**
1. Ratings ≥ 4 express preference.
2. A user's past interactions predict their future interactions.
3. MovieLens timestamps order a user's ratings, but are unreliable within the same second.

---

## 5.3 Functional requirements

### 5.3.1 Research pipeline

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Download each dataset from its official source and verify its checksum; reject corrupted files | ✅ |
| FR-2 | Parse every dataset into one shared schema (user, item, rating, timestamp; item and user attributes) and validate it (no duplicates, no unknown ids) | ✅ |
| FR-3 | Produce report-ready exploratory statistics and figures | ✅ |
| FR-4 | Create a reproducible per-user chronological train/validation/test split, verify that it has no leakage, and record its fingerprint | ✅ |
| FR-5 | Provide one common interface (`fit`, `score`, `recommend`) for all algorithms | ✅ |
| FR-6 | Implement the classic baselines: MostPop, UserKNN, ItemKNN, EASE, BPR-MF | ✅ |
| FR-7 | Implement the deep models: NeuMF, Mult-VAE, LightGCN, SASRec; and MF-SVD and iALS | ✅ |
| FR-8 | Evaluate every model with the same full-ranking evaluator: HR, Precision, Recall, NDCG, MRR, MAP at K ∈ {5, 10, 20}, plus coverage, novelty, popularity, diversity and exposure Gini | ✅ |
| FR-9 | Store per-user metrics for every run (for significance tests and cold-start analysis) | ✅ |
| FR-10 | Log every run (parameters, metrics, training curve, split fingerprint, artifacts) to MLflow | ✅ |
| FR-11 | Generate result tables (Markdown + CSV) for the report from the logged runs | ✅ |
| FR-12 | Tune every model with the same Optuna budget on the validation set | ✅ (LightGCN and SASRec keep their defaults: one trial already exceeds the budget on the laptop CPU) |
| FR-13 | Run paired significance tests between models | ✅ (Wilcoxon signed-rank, Holm-corrected) |
| FR-14 | Build the hybrid recommender: multi-model candidates → LightGBM ranker → MMR diversity re-ranking | ✅ |
| FR-15 | Repeat the benchmark on Amazon Reviews 2023 | ✅ (Video Games) |
| FR-16 | Export trained models as lightweight serving artifacts (NumPy arrays, LightGBM) | ✅ (SASRec re-implemented in NumPy, so ONNX was not needed) |

### 5.3.2 Web application and API

| ID | Requirement | Status |
|---|---|---|
| FR-20 | Users can sign up, log in and log out | ✅ email + password accounts built into the API (scrypt-hashed passwords, signed tokens, rate-limited login, password change, sign out everywhere); Supabase Auth as an alternative mode |
| FR-21 | A new user picks favourite genres and rates 10–15 seed movies during onboarding | ✅ two-step onboarding page (genres, then seed ratings; ≥ 3 required, 10 suggested) |
| FR-22 | The home page shows personalized rows (Top Picks, Because you liked X, Trending, Hidden Gems) | ✅ home page with all four rows plus a diversity slider |
| FR-23 | Rating a movie updates the next recommendations immediately, without retraining | ✅ (the user's history is folded in at request time) |
| FR-24 | Each recommendation shows a human-readable reason | ✅ (`reason` field on every recommendation) |
| FR-25 | A movie page shows details, a rating widget and similar movies (by collaborative filtering and by content) | ✅ movie page; similar films by viewers (CF) or by story and themes (text embeddings) |
| FR-26 | Keyword and natural-language (semantic) search | ✅ title search and "describe it" semantic search (`/search/semantic`) |
| FR-27 | Model Lab: any user's recommendations from each algorithm side by side, with a diversity slider | ✅ Model Lab page: you or a MovieLens viewer, six models side by side, overlap heatmap, MMR slider |
| FR-28 | Results dashboard that visualizes the offline experiments | ✅ Results page: both datasets, charts with table view, report figures |
| FR-29 | Log impressions and clicks for later analysis | ✅ (impressions logged automatically; clicks via `/events`) |
| FR-30 | Compare recommendation variants with real users (online A/B test) | ✅ stable hash-based assignment; impressions and clicks logged per group; click-through rate per group with a two-proportion z-test on the operators' page |
| FR-31 | Users can download all their data and delete their account | ✅ `/me/export`, `/me/delete` (password-confirmed) |

---

## 5.4 Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Correctness | Metric implementations match hand-computed reference values; split leakage checks run automatically; every model's math is checked against an independent computation (✅ 144 tests: 115 ML + 29 API, run in CI with the web app's lint, type check and build) |
| NFR-2 | Reproducibility | The same configuration and seeds reproduce the same split (identical fingerprint) and, for deterministic models, identical metrics (✅) |
| NFR-3 | Fairness of comparison | Every model uses the same split, evaluator and tuning budget (✅; see FR-12 for the two untuned models) |
| NFR-4 | Performance (offline) | Classic models train in under 1 minute and deep models within the free GPU quota on ML-1M |
| NFR-5 | Performance (online) | `/recommendations` responds within 200 ms at the 95th percentile on the free 2-vCPU server (measured locally at about 36 ms per request on the development laptop (median 34 ms in the smoke test); to be re-measured on the server) |
| NFR-6 | Scalability | Evaluation and UserKNN work in user batches, so memory stays bounded as the number of users grows (✅) |
| NFR-7 | Security | Passwords hashed with scrypt; signed, revocable, expiring tokens; login rate limiting; no account enumeration; row-level security on hosted Postgres; security headers; secrets only in environment variables (✅ tested) |
| NFR-8 | Privacy | No personal data beyond email and ratings; datasets are not redistributed (✅ the public deployment ships no MovieLens user histories) |
| NFR-9 | Usability | Responsive UI (mobile and desktop); recommendations always show a reason (✅ checked at 390 px and 1440 px in a browser test) |
| NFR-10 | Maintainability | Typed Python, linting (ruff), unit tests, CI on every push, one config file per dataset (✅) |
| NFR-11 | Portability | Backend in a Docker container; the serving image does not contain PyTorch, to stay small (✅ the Hugging Face Space image builds and runs: 1.05 GB with no PyTorch, ready in 10 s, 444 MB of memory, smoke test 10/10) |
| NFR-12 | Availability | The free tier sleeps when idle; mitigated by warming the service before demos and keeping a recorded demo video |

---

## 5.5 Hardware and software requirements

| | Minimum (development) | Used in this project |
|---|---|---|
| CPU | 4 cores | 8 logical cores |
| RAM | 8 GB | — |
| Disk | 5 GB free (environment ~2 GB, datasets ~1 GB) | 116 GB free |
| GPU | not required (Kaggle/Colab free GPU for deep models on large data) | none locally |
| OS | Windows 10/11, Linux or macOS | Windows 11 |
| Software | Python 3.12, uv, Git; Node.js 20+ for the frontend (Week 5); Docker (Week 4) | Python 3.12.10, uv 0.12.3, Node 24, Docker 29 |

---

## 5.6 Use cases (summary)

| ID | Actor | Use case | Main flow |
|---|---|---|---|
| UC-1 | Researcher | Run benchmark | `recsys benchmark -c ml-1m` → split loaded/verified → each model trained, validated, refit, tested → results logged and tables written |
| UC-2 | Researcher | Inspect experiments | open the MLflow UI → compare runs, curves and parameters |
| UC-3 | End user | Onboard | sign up → choose genres → rate seed movies → first recommendations shown |
| UC-4 | End user | Get recommendations | open home → personalized rows with reasons |
| UC-5 | End user | Rate a movie | rate → recommendations refresh immediately |
| UC-6 | End user | Find a movie | keyword or natural-language search → results → movie page |
| UC-7 | Examiner | Compare algorithms | Model Lab → pick a user → side-by-side lists, overlap, diversity slider |
