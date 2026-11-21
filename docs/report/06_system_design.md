# Chapter 6: System Design

> Draft (Week 4). The diagrams are written in Mermaid: they render on GitHub, and can be exported as
> images for the report at https://mermaid.live.

---

## 6.1 Architecture overview

Red Box has two halves that meet in one artefact, the **model bundle**.

| Half | What it does | Where it runs |
|---|---|---|
| Offline research pipeline | trains and evaluates every model, then packs the deployed model into a bundle | laptop CPU or a Kaggle GPU |
| Online application | serves recommendations from the bundle and stores what users do | Vercel (website) and Render (API) |

```mermaid
flowchart LR
    subgraph Offline["Offline (laptop / Kaggle)"]
        D[(MovieLens, Amazon<br/>raw files)] --> P[Prepare + split<br/>leakage-checked]
        P --> T[Train + tune<br/>12 models]
        T --> E[Evaluate<br/>full ranking]
        E --> M[(MLflow<br/>runs, metrics)]
        T --> H[Hybrid ranker<br/>LightGBM]
        H --> X[Export bundle]
    end
    X --> B[[Model bundle<br/>NumPy + LightGBM + items]]
    B -->|upload| HF[(Private Hugging Face dataset)]
    subgraph Online["Online"]
        U((User)) --> F[Next.js frontend<br/>Vercel]
        U -->|HTTPS + JWT| A[FastAPI<br/>Render, Docker, 2 GB]
        A --> S[(SQLite on a persistent disk<br/>accounts, ratings, watchlist, events)]
        A -. daily .-> W[Wikidata + Wikipedia<br/>new releases]
    end
    HF -->|first start| A
```

---

## 6.2 Offline pipeline (Weeks 0–3)

```mermaid
flowchart TB
    A[recsys download<br/>checksum-verified] --> B[recsys prepare<br/>shared schema + validation]
    B --> C[recsys eda<br/>statistics + figures]
    B --> D[recsys split<br/>per-user chronological 80/10/10]
    D --> E[recsys tune<br/>Optuna, validation only]
    E --> F[recsys benchmark<br/>fit on train, early stop on val,<br/>refit on train+val, test once]
    F --> G[(MLflow + saved models<br/>keyed by settings hash)]
    G --> H[recsys hybrid<br/>candidates + LambdaRank + MMR]
    G --> I[recsys export<br/>serving bundle]
    G --> J[recsys report<br/>tables for the report]
```

**Reproducibility guarantees**
- Every split carries a fingerprint, a hash of its exact contents.
- Every run carries a settings hash, logged to MLflow.
- A run with identical settings is reused instead of being repeated.

---

## 6.3 Serving architecture: how a recommendation is made

The API never retrains. A request rebuilds the user's history from the database, and every generator
scores the catalog from that history.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Frontend (Next.js)
    participant API as FastAPI
    participant DB as Database (SQLite on disk)
    participant EN as Engine (in memory)
    User->>FE: opens the home page
    FE->>API: GET /recommendations/feed (Bearer JWT)
    API->>API: verify token, get user id
    API->>DB: SELECT ratings of user ORDER BY created_at
    DB-->>API: liked movies (in order), all rated movies
    API->>EN: recommend(history, exclude=rated)
    EN->>EN: generators score all items:<br/>EASE, SASRec (NumPy), Markov, iALS fold-in, popularity
    EN->>EN: top-40 of each = candidates, about 40 features per pair
    EN->>EN: LightGBM ranks candidates, optional MMR diversity
    EN-->>API: top-k + reasons ("Because you liked …")
    API-->>FE: JSON rows (top picks, because you liked, hidden gems, trending)
    API--)DB: background: log impressions
    User->>FE: rates a movie ★★★★★
    FE->>API: PUT /ratings/{movie_id}
    API->>DB: upsert rating
    Note over API,EN: next request already uses the new rating:<br/>no retraining, no cache to invalidate
```

**Latency budget (target: under 200 ms):**
- one EASE row product (3,458 × 3,458);
- one SASRec forward pass in NumPy (at most 200 positions);
- one 64 × 64 linear solve for iALS;
- LightGBM on about 150 rows.

---

## 6.4 Data model

The **movie catalog is part of the model bundle**, not the database. The catalog and the models can therefore never disagree, for example a movie that exists in the database but has no embedding. The database holds only user activity.

```mermaid
erDiagram
    AUTH_USERS ||--o| PROFILES : has
    AUTH_USERS ||--o{ RATINGS : gives
    AUTH_USERS ||--o{ WATCHLIST : saves
    AUTH_USERS ||--o{ EVENTS : generates
    BUNDLE_ITEMS ||--o{ RATINGS : "movie_id"
    BUNDLE_ITEMS ||--o{ WATCHLIST : "movie_id"
    BUNDLE_ITEMS ||--o{ EVENTS : "movie_id"

    AUTH_USERS {
        uuid id PK "account id"
        text email
    }
    PROFILES {
        uuid user_id PK, FK
        varchar display_name
        jsonb favorite_genres
        timestamptz onboarded_at
    }
    RATINGS {
        uuid user_id PK, FK
        int movie_id PK
        float rating "0.5 to 5"
        timestamptz created_at "order of history"
    }
    WATCHLIST {
        uuid user_id PK, FK
        int movie_id PK
        timestamptz added_at
    }
    EVENTS {
        bigint id PK
        uuid user_id FK
        int movie_id
        varchar type "impression, click, watchlist, rate"
        varchar source_model
        int position
    }
    BUNDLE_ITEMS {
        int item_idx PK "row in model matrices"
        int movie_id UK "MovieLens id"
        text title
        smallint year
        text_array genres
        float popularity
    }
```

**Security:** row-level security on every table (`auth.uid() = user_id`) backs up the API's own checks (`supabase/migrations/`).

---

## 6.5 Data flow diagrams

**Level 0 (context)**

```mermaid
flowchart LR
    U[User] -- ratings, genres, searches --> S((Red Box))
    S -- recommendations + reasons, movie pages --> U
    DS[(Public datasets)] -- interactions --> S
    TM[TMDB] -- posters, overviews --> S
    R[Researcher] -- configs --> S
    S -- metrics, tables, figures --> R
```

**Level 1**

```mermaid
flowchart TB
    U[User] -->|rating| P1[1. Record activity]
    P1 --> D1[(Ratings / events)]
    U -->|page request| P2[2. Build history]
    D1 --> P2
    P2 --> P3[3. Generate candidates<br/>5 generators]
    B[(Model bundle)] --> P3
    P3 --> P4[4. Rank + diversify<br/>LightGBM, MMR]
    B --> P4
    P4 --> P5[5. Explain<br/>similar liked movie / genre]
    P5 -->|recommendations| U
    DS[(Datasets)] --> P6[6. Train, tune, evaluate]
    P6 --> D2[(MLflow)]
    P6 --> P7[7. Export]
    P7 --> B
```

---

## 6.6 Class diagram (core of the `recsys` package)

```mermaid
classDiagram
    class Recommender {
        <<abstract>>
        +params: dict
        +fit(data, validator) Recommender
        +score(users) ndarray
        +recommend(users, k, exclude) ndarray
        +refit_params() dict
    }
    class IterativeRecommender {
        <<abstract>>
        #_run_epoch(epoch) float
        #_snapshot() state
        #_restore(state)
        #_train(validator) early stopping
    }
    Recommender <|-- IterativeRecommender
    Recommender <|-- MostPop
    Recommender <|-- ItemKNN
    Recommender <|-- UserKNN
    Recommender <|-- EASE
    Recommender <|-- MarkovChain
    IterativeRecommender <|-- BPRMF
    IterativeRecommender <|-- IALS
    IterativeRecommender <|-- NeuMF
    IterativeRecommender <|-- MultVAE
    IterativeRecommender <|-- LightGCN
    IterativeRecommender <|-- SASRec

    class Split {
        +train, val, test: DataFrame
        +fingerprint() str
        +check() leakage
        +train_data(parts) TrainData
    }
    class Evaluator {
        +evaluate(model, history, targets, ks) EvalResult
    }
    class HistoryScorer {
        <<abstract>>
        +set_history(matrix, sequences)
        +score(users) ndarray
    }
    Recommender <|-- HistoryScorer
    HistoryScorer <|-- EASEScorer
    HistoryScorer <|-- SASRecScorer
    HistoryScorer <|-- IALSScorer
    HistoryScorer <|-- MarkovScorer
    HistoryScorer <|-- PopularityScorer
    class Engine {
        +recommend(history, exclude, k, lam, model)
        +similar(item, k, mode)
        +because_you_liked(history)
        +hidden_gems(history)
        +lab_compare(history)
    }
    Engine o-- HistoryScorer : 5 generators
    Engine o-- LightGBMRanker
    Evaluator ..> Recommender : scores
    Split ..> Evaluator : targets
```

---

## 6.7 Use cases

```mermaid
flowchart LR
    V([Visitor]) --- UC1[Browse and search movies]
    V --- UC2[Sign up / sign in]
    U([Signed-in user]) --- UC3[Onboarding: genres + seed ratings]
    U --- UC4[Get personalized feed]
    U --- UC5[Rate a movie / instant update]
    U --- UC6[Watchlist]
    U --- UC7[Adjust diversity]
    E([Examiner / researcher]) --- UC8[Model Lab: compare algorithms]
    E --- UC9[View offline results]
    E --- UC10[Reproduce experiments]
```

---

## 6.8 Deployment

```mermaid
flowchart LR
    subgraph Vercel
        FE[Next.js app]
    end
    subgraph Render["Render (Docker, Standard: 2 GB, 1 CPU)"]
        API[FastAPI + engine]
        DISK[(Persistent disk:<br/>SQLite, bundle, posters)]
    end
    Browser --> FE
    Browser -->|HTTPS + JWT| API
    API --> DISK
    HFD[(Private Hugging Face dataset:<br/>model bundle)] -->|first start| DISK
    GH[GitHub repo] -. build .-> API
    GH -. build .-> FE
```

- **Live:** https://redbox-recommender.vercel.app, with the API at https://redbox-api.onrender.com.
- **Why this layout.** Hugging Face now charges for Docker Spaces, and free hosts without a card offer too
  little memory for the engine (about 1 GB), so the API runs on a paid Render Standard instance. Accounts live in
  SQLite on the service's persistent disk. Postgres (`supabase/migrations/`) remains supported.
- **The bundle** is too large for git and contains model weights derived from MovieLens, so it lives in a private
  Hugging Face dataset. `deploy/render/start.py` downloads it onto the disk on first start.

---

## 6.9 Key design decisions

| Decision | Alternative considered | Why |
|---|---|---|
| Catalog shipped inside the model bundle | catalog table in the database | the catalog always matches the models; no sync step |
| Torch-free serving: SASRec re-implemented in NumPy | PyTorch or ONNX Runtime in the API | small image and fast cold start on a free tier; a unit test proves identical outputs |
| Semantic search: MiniLM through ONNX Runtime, cosine over in-memory vectors | sentence-transformers (PyTorch); pgvector in the database | no PyTorch in the image; 3,458 × 384 vectors are 5 MB, so an in-memory dot product (~13 ms) beats a database round trip |
| Viewer keywords from MovieLens-32M tags as the movie text | TMDB overviews only | free and needs no API key; TMDB plots are added when a key is available |
| Public bundle ships aggregate item statistics, not user histories | publish the full bundle | the MovieLens licence forbids redistributing the ratings; a test shows recommendations are identical |
| Serving hybrid uses only history-based generators | the offline hybrid with LightGCN | new users and new ratings are handled instantly; LightGCN would need retraining |
| Serving ranker trained on features from the serving scorers | reuse the offline ranker | identical feature code for training and serving, so no train/serve skew |
| Recommendations recomputed per request | per-user cache | the next request always reflects the latest rating; computing is cheap (tens of ms) |
| Settings-hash caching of runs and models | rerun everything per report | hours of CPU training are never repeated for identical settings |
| Row-level security + API checks | API checks only | defence in depth if a client reaches the database directly |
