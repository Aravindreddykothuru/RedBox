# Chapter 4: Problem Statement and Objectives

## 4.1 Problem statement

> Given only the record of which items each user liked and when, recommend to each user a short ranked list
> of items they are likely to like next. Do so accurately and with reasonable variety. Determine, under a
> fair and reproducible protocol, which collaborative-filtering techniques achieve this best. Deliver the
> result as a system that serves new users instantly.

Formally: let $U$ be the users, $I$ the items and $X \in \{0,1\}^{|U|\times|I|}$ the implicit-feedback
matrix ($x_{ui}=1$ if user $u$ rated item $i$ at least 4 stars). Each interaction also has a timestamp.
For each user, a model must produce a ranking of all items the user has not interacted with, so that the
items the user goes on to like appear as high as possible (measured by NDCG@10 and the other metrics of
Section 7.3).

**Why it is hard**
- **Sparsity.** MovieLens-1M is 95.5% empty, and the Amazon data is far sparser: a median of 6 liked items
  per user.
- **Long tail.** The 20% most popular movies receive 65% of all ratings (Gini 0.63). Popularity alone is a
  strong but useless baseline.
- **Cold start.** New users arrive with no history and must be served immediately.
- **Contested evidence.** Published comparisons disagree on whether deep models beat tuned classic methods.
- **Limited resources.** The project is restricted to free tools and a laptop CPU.

## 4.2 Objectives

| # | Objective | Status |
|---|---|---|
| O1 | Build a reproducible data and evaluation pipeline: chronological splits, full-ranking metrics, no data leakage | Achieved (Chapter 7; leakage and metric tests in Chapter 10) |
| O2 | Implement and tune 11 CF algorithms across the neighbourhood, latent-factor and deep families | Achieved; LightGCN and SASRec use their papers' settings, because one training run exceeds the tuning budget on a CPU |
| O3 | Compare them on accuracy and beyond-accuracy metrics, with significance tests | Achieved (Sections 11.2–11.6) |
| O4 | Propose a hybrid two-stage recommender and test whether it beats the best single model | Achieved: +9.0% / +8.3% NDCG@10, p < 10⁻²⁶ |
| O5 | Handle cold-start users (onboarding and instant fold-in) and give content-based access to items | Achieved for users; items get content similarity and semantic search. A learned content→CF mapping for brand-new items is future work |
| O6 | Deploy a full-stack web app with real-time personalisation and explanations | Built and tested end to end; deployment scripts are ready, and going live needs the owner's free accounts (Section 9.8) |
| O7 | Test whether the findings carry over to product recommendation (Amazon) | Achieved (Section 11.7) |

## 4.3 Research questions

- **RQ1:** Do deep models (NeuMF, Mult-VAE, LightGCN, SASRec) beat well-tuned classic CF (ItemKNN, iALS,
  EASE)?
- **RQ2:** Does the hybrid two-stage ranker improve over the best single model, and which signals matter
  most?
- **RQ3:** How does each model trade accuracy against diversity, novelty and coverage, and how do models
  serve users with little history?
- **RQ4:** Do the conclusions hold on product data (Amazon) as well as on movies?

## 4.4 Scope

**In scope**
- **Data:** implicit feedback derived from explicit ratings (rating ≥ 4 = liked), plus a side experiment
  on rating prediction (RMSE/MAE).
- **Datasets:** two public datasets; models trained from our own implementations.
- **Application:** a web application serving the MovieLens-1M models, with accounts, ratings, a
  watchlist, explanations, semantic search, a Model Lab and a results dashboard.

**Out of scope (and why)**
- **Models trained on MovieLens-32M for the app.** They would add recent films, but the CPU and memory
  budget did not allow them. The plan's 4-week cut line serves the ML-1M models instead.
- **Online A/B testing with real users.** It needs real traffic. The system logs impressions and clicks so
  that this is possible later.
- **Multi-seed repetition of every experiment.** It is CPU-bound and is listed as a threat to validity
  (Section 11.11).
- **Purely content-based recommendation for brand-new items.** The content side currently supports search
  and similarity.
