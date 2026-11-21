# Abstract

Recommender systems help people choose from catalogues far too large to browse. Collaborative filtering
(CF) learns those recommendations from user–item interactions alone. Many deep-learning models have been
proposed for it, but reproducibility studies report that well-tuned classic methods often match them. This
project asks which CF techniques work best for personalised movie and product recommendation when every
model is treated equally, and turns the answer into a working product.

**The comparison.** We implemented eleven algorithms behind one interface:
- neighbourhood methods (UserKNN, ItemKNN);
- latent-factor models (BPR-MF, iALS);
- a linear autoencoder (EASE);
- deep models (NeuMF, Mult-VAE);
- a graph neural network (LightGCN);
- sequential models (a Markov chain and the SASRec Transformer).

Every model was evaluated under one leakage-free protocol: a per-user chronological split, ranking of the
full catalogue, the same Optuna tuning budget and paired significance tests. The datasets were MovieLens-1M
and Amazon Reviews 2023 (Video Games).

**Findings.**
- **Deep vs. classic.** A tuned EASE model, trained in two seconds, significantly beats every non-sequential
  deep model on both datasets.
- **The MovieLens rating-session artifact.** SASRec leads on MovieLens, but an order ablation and a
  session analysis trace most of that lead to an artifact of how MovieLens collected ratings: trained on
  shuffled histories, SASRec loses 61% of its NDCG@10.
- **The proposed hybrid.** Red Box is a two-stage system. Several models propose candidates, a LightGBM
  LambdaRank model ranks them using the models' scores together with item and user features, and MMR
  re-ranking adds diversity. It improves NDCG@10 over
  the best single model by 9.0% on MovieLens (0.1790) and by 8.3% on Amazon (0.0919), with p < 10⁻²⁶.

**The deployed system.** Red Box runs as a full-stack web application: a Next.js frontend, a FastAPI
service that runs without PyTorch, and Supabase for accounts. New users get personalised, explained
recommendations after rating a few films, and every rating changes the next list in about 35 ms without
retraining. Natural-language search uses sentence embeddings of viewer keywords. A Model Lab shows each
algorithm's recommendations side by side, and a dashboard presents the experiments.

**Keywords:** collaborative filtering, recommender systems, learning to rank, sequential recommendation,
evaluation methodology, hybrid recommender.
