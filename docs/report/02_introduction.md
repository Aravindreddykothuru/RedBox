# Chapter 2: Introduction

## 2.1 Background

Streaming services, online shops and news sites offer far more items than anyone can look through. A
recommender system narrows that choice: from what a person and millions of others have done, it predicts
what this person is likely to want next. Recommenders drive a large share of what is watched and bought
online, which makes them one of the most widely deployed applications of machine learning.

**Collaborative filtering (CF)** is the core technique. It needs no description of the items, only the
record of who interacted with what. Its central idea fits in one sentence: *people who agreed in the past
tend to agree in the future.* Over three decades CF has grown through several families:
1. **Neighbourhood methods:** "users like you also liked…" and "items similar to this one".
2. **Latent-factor models:** users and items as points in a hidden "taste space".
3. **Neural networks:** autoencoders and graph neural networks.
4. **Sequential models:** these read a person's history in order and predict the next item.

## 2.2 Motivation

Three observations motivated this project.

1. **Is the progress real?** Each new deep model reports large gains. Yet independent reproducibility
   studies (Ferrari Dacrema et al., 2019; Rendle et al., 2020) found that many of those gains disappear
   against properly tuned simple baselines. A student, or a company, choosing a technique needs a comparison
   in which every model is treated equally.
2. **Accuracy is not the whole story.** A system that recommends only blockbusters can look accurate while
   being useless for discovery. Coverage, novelty and diversity matter to users.
3. **Research rarely becomes a product.** Offline experiments end with a results table. Real users bring
   problems a table hides:
   - new users with no history;
   - the need for instant updates;
   - the need for explanations;
   - limited computing resources.

## 2.3 What this project does

The project has two halves that inform each other.

- **A controlled comparison:**
  - eleven CF algorithms, from MostPop to a Transformer;
  - two datasets, MovieLens-1M (movies) and Amazon Reviews 2023 Video Games (products);
  - one protocol: chronological split, full-catalogue ranking, equal tuning budget, significance tests.
- **Red Box, a deployed hybrid recommender:**
  - several of those models propose candidates, a learning-to-rank model orders them, and an optional
    diversity step re-ranks the list;
  - it is served through a web application in which people sign up, rate a few films and immediately get
    explained recommendations.

The whole project runs on free tools and a 4-core laptop CPU, without a GPU. That constraint shaped many
design choices, and the report notes them.

## 2.4 Contributions

1. **A reproducible benchmark of eleven CF algorithms.**
   - Both datasets use one leakage-checked protocol.
   - Each run is tracked with its exact settings and split fingerprint, so every number in this report can
     be traced to a logged run.
2. **Evidence on the deep-versus-classic question.** A tuned linear autoencoder (EASE) beats every
   non-sequential deep model on both datasets, in line with the reproducibility literature.
3. **A diagnosis of the MovieLens "rating-session" artifact.** Sequential models look excellent on MovieLens
   largely because users rated films in bulk, in the order the website showed them. We show this three ways:
   - a parameter-free Markov chain beats every taste model;
   - analysing users by when they returned reverses the ranking;
   - SASRec trained on shuffled histories loses 61% of its accuracy.
   The Amazon data, which lacks the artifact, confirms the diagnosis.
4. **Red Box, a two-stage hybrid.**
   - It beats the best single model by 9.0% (MovieLens) and 8.3% (Amazon), both highly significant.
   - Its MMR diversity step gives a smooth, user-controllable trade-off between accuracy and variety.
5. **A torch-free, real-time serving design.**
   - The models are exported to NumPy, including a re-implementation of SASRec that is tested against the
     PyTorch original.
   - The API computes recommendations from the user's current history on every request, about 35 ms, so a
     new rating changes the very next list.
   - A test guarantees that the ranker sees identical features in training and in serving.
6. **A complete web application.** It covers:
   - onboarding;
   - an explained, personalised home feed;
   - natural-language search over viewer keywords;
   - movie pages with collaborative and content-based "similar films";
   - a Model Lab comparing algorithms side by side;
   - a results dashboard.

## 2.5 Organisation of the report

| Chapter | Contents |
|---|---|
| 3. Literature Survey | The CF families, evaluation studies, and the gap this project addresses |
| 4. Problem Statement and Objectives | The problem, objectives, research questions and scope |
| 5. Software Requirements Specification | Functional and non-functional requirements |
| 6. System Design | Architecture, data model, data flows, class and deployment diagrams |
| 7. Methodology | Data preparation, evaluation metrics and every algorithm, with equations |
| 8. Proposed System | The Red Box hybrid, its serving variant, cold start and the content side |
| 9. Implementation | Technology, code structure, the API, the web application and deployment |
| 10. Testing | Test strategy, test cases and their results |
| 11. Results and Discussion | All experiments, answering the four research questions |
| 12. Conclusion and Future Scope | Findings, limitations and next steps |
| 13. References | Works cited (IEEE style) |
