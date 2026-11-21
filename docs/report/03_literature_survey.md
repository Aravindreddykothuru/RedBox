# Chapter 3: Literature Survey

**Project:** Collaborative Filtering for Personalized Movie/Product Recommendations using ML/DL Techniques

The survey follows how collaborative filtering (CF) developed over time:

1. **Neighbourhood methods:** "people like you" and "items like this".
2. **Latent-factor models:** hidden taste vectors.
3. **Deep and graph models:** neural networks, autoencoders, graph convolution.
4. **Sequential models:** the order in which a user rates items.
5. **Evaluation and reproducibility studies:** whether the progress is real.
6. **Ranking and diversity techniques:** used to build our hybrid system.

The last column states how each paper is used in this project.

> Status: draft. Before submission, check every reference against the original paper (DOI or official PDF) and convert it to the IEEE format your department requires.

---

## 3.1 Comparison table

| # | Paper (venue, year) | Family | Core idea | Strength | Limitation / gap | Used in our project as |
|---|---|---|---|---|---|---|
| 1 | Sarwar, Karypis, Konstan, Riedl. *Item-based collaborative filtering recommendation algorithms* (WWW 2001) | Neighbourhood | Recommend items similar to those the user liked, with item–item similarity computed from co-ratings | Simple, explainable ("similar to X"), item similarities can be precomputed | Struggles with sparse data; similarities are not learned for the task | **ItemKNN** baseline; "Because you liked X" explanations |
| 2 | Koren, Bell, Volinsky. *Matrix factorization techniques for recommender systems* (IEEE Computer 2009) | Latent factor | Represent users and items as low-dimensional vectors; rating ≈ dot product + biases | Handles sparsity well; compact; won the Netflix Prize era | Built around explicit ratings and RMSE, not top-K ranking | **MF-SVD** for the rating-prediction experiment (RMSE/MAE) |
| 3 | Hu, Koren, Volinsky. *Collaborative filtering for implicit feedback datasets* (ICDM 2008) | Latent factor | Treat interactions as binary preference with confidence weights; solve with alternating least squares | Uses all unobserved items as weak negatives; scales well | No notion of order; linear interactions only | **iALS** model |
| 4 | Rendle, Freudenthaler, Gantner, Schmidt-Thieme. *BPR: Bayesian personalized ranking from implicit feedback* (UAI 2009) | Latent factor / learning-to-rank | Pairwise loss: a liked item should score higher than a non-liked one | Optimizes ranking directly; model-agnostic loss | Uniform negative sampling is inefficient | **BPR-MF** model; BPR loss also trains LightGCN |
| 5 | Sedhain, Menon, Sanner, Xie. *AutoRec: Autoencoders meet collaborative filtering* (WWW 2015) | Deep (autoencoder) | Reconstruct a user's or item's rating vector through a bottleneck network | First simple neural CF; non-linear | Explicit ratings; superseded by Mult-VAE for top-K | Background for Mult-VAE |
| 6 | Covington, Adams, Sargin. *Deep neural networks for YouTube recommendations* (RecSys 2016) | Industrial system | Two stages: candidate generation, then a ranking model with rich features | Scales to billions; the standard industry architecture | Needs large data and features; not a single model | Blueprint for our **two-stage hybrid** |
| 7 | He, Liao, Zhang, Nie, Hu, Chua. *Neural collaborative filtering* (WWW 2017) | Deep | Replace the dot product with a neural network; NeuMF fuses generalized MF and an MLP | Popularized deep CF; flexible | Later shown that a well-tuned dot product can match or beat it (#15) | **NeuMF** model |
| 8 | Liang, Krishnan, Hoffman, Jebara. *Variational autoencoders for collaborative filtering* (WWW 2018) | Deep (generative) | Variational autoencoder with a multinomial likelihood over a user's whole interaction vector | Strong top-K accuracy; scores new users without retraining | Needs tuning of the KL annealing; whole-catalog output layer | **Mult-VAE** model |
| 9 | Kang, McAuley. *Self-attentive sequential recommendation* (ICDM 2018) | Sequential (Transformer) | Self-attention over the user's ordered history predicts the next item | Captures order and recency; efficient | Assumes timestamps reflect real order (our EDA: 77% ties in ML-1M) | **SASRec** model; order ablation |
| 10 | Steck. *Embarrassingly shallow autoencoders for sparse data* (WWW 2019) | Linear autoencoder | Closed-form item–item weight matrix with a zero diagonal (EASE^R) | Very strong yet simple; one matrix inversion; instant for new users | Memory grows with the square of the number of items | **EASE** model; candidate generator in the hybrid |
| 11 | Wang, He, Wang, Feng, Chua. *Neural graph collaborative filtering* (SIGIR 2019) | Graph neural network | Propagate embeddings over the user–item graph to capture high-order connectivity | Uses multi-hop collaborative signal | Heavy non-linear layers; simplified by LightGCN | Background for LightGCN |
| 12 | Sun et al. *BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer* (CIKM 2019) | Sequential (Transformer) | Bidirectional transformer trained with masked-item prediction | Uses context on both sides of an item | Gains over SASRec are disputed in later replications | Stretch goal; related work |
| 13 | Ferrari Dacrema, Cremonesi, Jannach. *Are we really making much progress? A worrying analysis of recent neural recommendation approaches* (RecSys 2019) | Reproducibility study | Re-ran published neural recommenders against tuned simple baselines | Exposed weak baselines and poor reproducibility | Covers specific papers only | Motivates **RQ1** and our equal tuning budget |
| 14 | He, Deng, Wang, Li, Zhang, Wang. *LightGCN: Simplifying and powering graph convolution network for recommendation* (SIGIR 2020) | Graph neural network | Keep only neighbourhood aggregation; drop feature transforms and non-linearities | Simpler and more accurate than NGCF | Transductive: a new user needs fold-in | **LightGCN** model; main candidate generator |
| 15 | Rendle, Krichene, Zhang, Anderson. *Neural collaborative filtering vs. matrix factorization revisited* (RecSys 2020) | Reproducibility study | Shows a properly tuned dot-product model matches or beats NCF's MLP | Clarifies that an MLP struggles to learn a dot product | Specific to the NCF setting | Supports **RQ1** discussion |
| 16 | Krichene, Rendle. *On sampled metrics for item recommendation* (KDD 2020) | Evaluation methodology | Shows that metrics computed on a small sample of items can change model rankings | Gives a clear rule: use full ranking | – | Our **full-catalog evaluation** protocol |
| 17 | Harper, Konstan. *The MovieLens datasets: History and context* (ACM TiiS 2015) | Dataset | Documents how MovieLens data was collected and its biases | Explains data quirks (bulk rating, filtering) | – | Dataset description and limitations chapter |
| 18 | Hou, Li, He, Yan, Chen, McAuley. *Bridging language and items for retrieval and recommendation* (2024) | Dataset | Introduces Amazon Reviews 2023 with standard benchmark splits | Large, recent, many product categories | Very sparse; needs filtering (5-core) | **Amazon** product benchmark (RQ4) |
| 19 | Ke et al. *LightGBM: A highly efficient gradient boosting decision tree* (NeurIPS 2017); Burges. *From RankNet to LambdaRank to LambdaMART: An overview* (Microsoft Research Tech. Report 2010) | Learning-to-rank | Gradient-boosted trees trained with a ranking (NDCG-aware) objective | Fast on CPU; handles mixed features; strong ranker | Needs a candidate set and labelled data | **Stage 2 ranker** of the hybrid |
| 20 | Carbonell, Goldstein. *The use of MMR, diversity-based reranking for reordering documents and producing summaries* (SIGIR 1998) | Diversity | Greedily pick items balancing relevance against similarity to items already chosen | Simple, one tunable trade-off parameter λ | Greedy; needs an item-similarity measure | **Stage 3 diversity re-ranking** |

---

## 3.2 Research gap (what this project adds)

1. **Conflicting evidence on deep vs. classic CF.** Papers proposing deep models (#7, #8, #11, #14) report large gains. Reproducibility studies (#13, #15) show that well-tuned simple baselines often match them. We compare **11 algorithms under one protocol** with:
   - one shared chronological split
   - full-catalog ranking (#16)
   - an equal tuning budget for every model
   - significance tests
2. **Accuracy-only evaluation.** Most papers report only Recall and NDCG. Our EDA shows a strong long tail: the top 20% of movies receive 65% of all ratings (Gini 0.63). Models that only push blockbusters look good on accuracy, so we also report **coverage, novelty, diversity and popularity bias**.
3. **Single-model systems.** Academic CF papers usually propose one model, while industry systems combine many (#6). We test whether a **two-stage hybrid** beats the best single model, and which signals matter for it.
4. **Movies only.** Many CF studies use only MovieLens. We repeat the comparison on **Amazon product data** (#18) to test whether the conclusions generalize.
5. **From research to a usable system.** We deploy the result as a web app that updates recommendations instantly for new users and gives explanations.

---

## 3.3 Short summaries by family

**Neighbourhood methods (#1).** These compute similarity between users or between items from co-rated items and recommend based on the nearest neighbours. They are intuitive and explainable, but on sparse data most pairs share very few ratings, so the similarities are noisy.

**Latent-factor models (#2–#4).** These map users and items into a shared low-dimensional space where closeness means preference. They differ in the data they assume and the loss they optimize:
- MF-SVD minimizes rating error on explicit ratings.
- iALS weights implicit interactions by confidence.
- BPR directly optimizes pairwise ranking.

**Deep models (#5, #7, #8).** These add non-linearity:
- NeuMF learns the user–item interaction function with an MLP.
- Autoencoders (AutoRec, Mult-VAE) reconstruct a user's full interaction vector, which lets them score new users without retraining.

**Graph models (#11, #14).** These treat the data as a bipartite user–item graph and propagate embeddings along its edges, so a user's representation also reflects items liked by similar users (multi-hop signal). LightGCN showed that the simple propagation alone is what makes these models work.

**Sequential models (#9, #12).** These read a user's history as an ordered sequence and predict the next item with self-attention.

**Evaluation (#13, #15, #16).** These papers question whether published gains are real. Their main points:
- baselines must be properly tuned
- evaluation should rank the full catalog, not a sample of items

**Ranking and diversity (#6, #19, #20).** Industry systems separate retrieval (candidate generation) from ranking with rich features, and re-rank the final list for diversity. We use all three ideas in our hybrid.

---

## 3.4 Additional works used in the system

| # | Work | Used in our project as |
|---|---|---|
| 21 | Reimers, Gurevych. *Sentence-BERT: Sentence embeddings using Siamese BERT-networks* (EMNLP-IJCNLP 2019) | The sentence-embedding approach behind semantic search and content similarity |
| 22 | Wang, Wei, Dong, Bao, Yang, Zhou. *MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers* (NeurIPS 2020) | The small encoder (all-MiniLM-L6-v2) that embeds movie texts and search queries |
| 23 | Fan, Ji, Zhang, Sun. *Our model achieves excellent performance on MovieLens: What does it mean?* (ACM TOIS, 2024) | Explains why sequential models are inflated by MovieLens' rating sessions; supports Section 11.4 |
| 24 | Akiba, Sano, Yanase, Ohta, Koyama. *Optuna: A next-generation hyperparameter optimization framework* (KDD 2019) | Equal-budget hyperparameter tuning (TPE sampler, median pruning) |
