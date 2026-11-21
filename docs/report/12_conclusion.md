# Chapter 12: Conclusion and Future Scope

## 12.1 Conclusion

This project set out to find which collaborative-filtering techniques work best for personalised
recommendation when every model is treated equally, and to turn the answer into a usable product.

**Comparison.** Eleven algorithms, implemented from their papers, were compared on MovieLens-1M and Amazon
Video Games. All used one leakage-free protocol: chronological split, full-catalogue ranking, equal tuning
budget, significance tests.

**What we found**
1. **Tuned classic methods are hard to beat.** EASE, a linear autoencoder solved in closed form in two
   seconds, is significantly more accurate than every non-sequential deep model on both datasets. That
   includes LightGCN, a graph neural network that takes 40 minutes to train. This independently confirms
   the reproducibility studies that motivated the project.
2. **Order matters, but the data decide how much.** On MovieLens the sequential models lead by a wide
   margin. Three analyses show that most of this lead comes from how MovieLens collected ratings:
   - a parameter-free Markov chain beats all taste models;
   - the ranking reverses for users who return on a later day;
   - SASRec trained on shuffled histories loses 61% of its accuracy and falls below EASE.

   On Amazon purchases, which have no such sessions, the Markov chain's advantage disappears. SASRec keeps a
   smaller but genuine lead.
3. **Combining models helps.** The proposed Red Box hybrid improves NDCG@10 over the best single model
   on both datasets:
   - MovieLens: 0.1790, +9.0%;
   - Amazon: 0.0919, +8.3%;
   - significance: p < 10⁻²⁶ on both.

   Its ranker learns when to trust the sequential signal and when to fall back on long-term taste. MMR
   re-ranking gives a smooth, controllable trade-off between accuracy and variety.
4. **Research results can be served cheaply.** Exported to NumPy, the hybrid scores a user from their current
   history in about 35 ms on a laptop CPU, without PyTorch and without retraining. Its accuracy is the same
   as the research version.

**What was built.** Red Box is a complete web application:
- onboarding that serves new users after a few ratings;
- a home feed whose every item explains itself;
- natural-language search built from viewer keywords and sentence embeddings;
- movie pages with collaborative and content-based "similar films";
- a Model Lab that compares the algorithms side by side;
- a results dashboard.

It was tested from unit level (197 automated tests) to a full browser journey, and it is ready to deploy on
free tiers.

**For a practitioner, the lessons are:**
- tune the simple baselines before reaching for deep models;
- check what the evaluation data rewards;
- combine complementary models with a learned ranker.

## 12.2 Limitations

- **One split per dataset.** Results are from a single chronological split (one seed). Repeating with
  several split seeds would give confidence intervals.
- **Unequal tuning in practice.** SASRec and LightGCN could not be tuned within the CPU budget. This can
  only understate them.
- **Implicit-feedback threshold.** "Liked" means a rating of 4 or more. Other thresholds were not tested.
- **The models' own catalogue ends in 2023.**
  - The app serves models trained on MovieLens-32M ratings from 2018 to October 2023 (8,487 films). Newer
    films reach it as new releases (Section 8.5), recommended from their story and their first ratings
    until retraining makes them regular films.
  - To fit laptop memory, its ranker is trained on a sample of 15,000 users, and the served hybrid is
    evaluated on another 15,000 (Section 11.12).
- **No online evaluation yet.** All accuracy figures are offline. The app has a built-in A/B test with
  live click-through metrics, but it has not yet run with real users.
- **New items live in one model's space.** A film with no ratings is placed in the iALS factor space by a
  linear mapping from its description (Section 11.14), and its first ratings refine that guess (Section
  11.15). That recovers part of the signal, not all of it, and the hybrid's other generators still cannot
  score the film until retraining. The warm-up uses only the app's raters, so with few users it moves
  slowly.
- **New releases depend on Wikipedia.** A film enters once its English Wikipedia article exists and, for
  English films, once enough language editions cover it, which can take days after release.

## 12.3 Future scope

1. **Learning from the app's users.**
   - Retrain the app's models as people rate: run `recsys import-app-ratings`, then
     `train_app_models.sh --refresh`. New releases with 20 likes then graduate into every model; running
     this on a schedule would close the loop completely.
   - Tune SASRec on the larger catalogue with a GPU from a free notebook service.
   - Fetch official posters and plots through TMDB.
2. **Better new-item cold start.** Replace the linear text-to-factor mapping with a small neural one, add
   cast, director and trailer features, and learn mappings into SASRec's and EASE's spaces too, so new
   films can enter the hybrid itself.
3. **Stronger statistics.** Repeat the benchmark over several split seeds and report confidence intervals.
   Tune SASRec and LightGCN on a GPU.
4. **Sequence-aware evaluation.** Build test sets that exclude same-session continuations, so that
   sequential models are judged on genuine next-item intent.
5. **Online learning from the logs.**
   - Use the logged impressions and clicks to evaluate rankers offline (counterfactual evaluation).
   - Retrain the ranker periodically.
   - Run A/B tests in the deployed app.
6. **Explanations users can act on.** The ranker's per-signal contributions are now shown (TreeSHAP,
   Section 8.4). The next step is to let users correct them ("less like this film"), and to test in the
   A/B framework whether explanations raise trust and click-through.
7. **Fairness and exposure.** Monitor how exposure is spread across popular and niche films over time, and
   add an exposure constraint to the re-ranking stage.
