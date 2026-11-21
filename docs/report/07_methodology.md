# Chapter 7: Methodology

> Draft.
> - **Part 1 (Week 1):** data preparation, evaluation protocol and metrics, and the neighbourhood, linear and latent-factor models (Sections 7.1–7.4.5).
> - **Part 2 (Week 2):** iALS, the deep-learning models (NeuMF, Mult-VAE, LightGCN, SASRec) and rating prediction (Sections 7.4.6–7.4.11).
>
> Every formula below is implemented in `ml/src/recsys/` and checked by unit tests (Section 7.6).

---

## 7.1 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | sets of users and items; $\lvert\mathcal{U}\rvert = m$, $\lvert\mathcal{I}\rvert = n$ |
| $X \in \{0,1\}^{m \times n}$ | interaction matrix; $x_{ui} = 1$ if user $u$ liked item $i$ in the training data |
| $\mathcal{I}_u$, $\mathcal{U}_i$ | items liked by user $u$; users who liked item $i$ |
| $s(u,i)$ | score a model gives item $i$ for user $u$ (higher = recommend first) |
| $R_u(K)$ | the top-$K$ list recommended to $u$ (items $u$ already knows are excluded) |
| $T_u$ | held-out target items of $u$ (validation or test) |

---

## 7.2 Data preparation

### 7.2.1 From ratings to implicit feedback
MovieLens ratings run from 1 to 5 stars. Top-K recommendation asks which items a user will *like*, so a rating of 4 or more is treated as a positive interaction ($x_{ui}=1$), and everything else as unobserved. This keeps 575,281 of the 1,000,209 ratings (57.5%). As an ablation (Week 3), every rating is treated as positive instead.

### 7.2.2 Per-user chronological split
Each user's liked items are sorted by time. The oldest 80% go to **training**, the next 10% to **validation** and the newest 10% to **test**. For a user with $n_u$ liked items:

$$n_u^{test} = \max\!\left(1, \left\lfloor 0.1\, n_u + 0.5 \right\rfloor\right), \quad n_u^{val} = \max\!\left(1, \left\lfloor 0.1\, n_u + 0.5 \right\rfloor\right), \quad n_u^{train} = n_u - n_u^{val} - n_u^{test}$$

Design decisions, each motivated by the exploratory analysis (Chapter 6):

1. **Per-user rather than global time cut.** A single global cut-off date would leave only 29.5% of users with any test data, because most users stopped rating in 2000. The per-user split keeps every user testable while still never training on a user's future.
2. **Seeded random tie-break.** 77% of ratings share the exact same second with another rating of the same user, because MovieLens users rated in bulk on rating pages. For 58.3% of users a split boundary falls inside such a tie, so the tie-break rule decides much of the test set. Ties are broken by a random key from a fixed seed: the split is identical on every run, but follows no hidden pattern such as item id or file order.
3. **Users with fewer than 3 liked items** cannot give one item to each part. They stay in training only, so their data still helps learn item relationships, but they are not evaluated (3 users). Two users with no liked items at all are removed.
4. **Cold items.** An item that never occurs in training cannot be scored by any collaborative model. Its validation and test interactions are removed (35 and 67 interactions, about 0.1%).

**Resulting split (MovieLens-1M, fingerprint `8d6a7a5a8a4c83ab`)**

| | Users | Items | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Count | 6,038 | 3,458 | 459,601 | 57,805 | 57,773 |
| Users evaluated | | | | 6,034 | 6,034 |

The split is saved with a fingerprint (a hash of its exact contents). Every experiment records that fingerprint, so any result can be traced to the exact data it used.

### 7.2.3 Experimental protocol
All models follow the same two-phase protocol:

1. **Fit on training data.** Models trained iteratively use the validation set for early stopping, monitored with NDCG@10. Hyperparameter tuning (Week 3) will also use only the validation set.
2. **Refit on training + validation** with the chosen settings, including the best number of epochs, then **evaluate once on the test set**.

The test set therefore never influences any modelling choice, so the test scores are an unbiased estimate of performance on future interactions.

---

## 7.3 Evaluation metrics

**Full ranking.** For every evaluated user the model scores all $n$ items, removes the items in the user's known history, and ranks the rest. Many papers rank each target against only ~100 sampled items, but Krichene & Rendle (2020) showed that sampled metrics can change which model appears best, so we always rank the full catalog.

Let $\mathrm{rel}_r = 1$ if the item at rank $r$ of $R_u(K)$ is in $T_u$, else $0$. All accuracy metrics are averaged over evaluated users.

| Metric | Definition | What it measures |
|---|---|---|
| Hit Rate | $HR@K = \mathbb{1}\left[\sum_{r=1}^{K}\mathrm{rel}_r > 0\right]$ | at least one correct item in the list |
| Precision | $P@K = \frac{1}{K}\sum_{r=1}^{K}\mathrm{rel}_r$ | share of the list that is correct |
| Recall | $R@K = \frac{1}{\lvert T_u\rvert}\sum_{r=1}^{K}\mathrm{rel}_r$ | share of the user's targets that were found |
| NDCG | $NDCG@K = \dfrac{\sum_{r=1}^{K}\mathrm{rel}_r/\log_2(r+1)}{\sum_{r=1}^{\min(K,\lvert T_u\rvert)}1/\log_2(r+1)}$ | correct items near the top count more; 1 = ideal order |
| MRR | $MRR@K = 1/r^{*}$, where $r^{*}$ is the rank of the first correct item (0 if none) | how soon the first correct item appears |
| MAP | $AP@K = \frac{1}{\min(K,\lvert T_u\rvert)}\sum_{r=1}^{K} P@r \cdot \mathrm{rel}_r$ | precision averaged over the positions of the hits |

**Beyond-accuracy metrics.** These describe *what* is recommended. The long-tail analysis (top 20% of movies receive 65% of ratings) showed that accuracy alone rewards recommending blockbusters.

| Metric | Definition | What it measures |
|---|---|---|
| Coverage | $\lvert \bigcup_u R_u(K)\rvert / n$ | share of the catalog that is ever recommended |
| Novelty | $\frac{1}{K}\sum_{i \in R_u(K)} -\log_2\!\left(\lvert\mathcal{U}_i\rvert/m\right)$ | how unknown the recommended items are (self-information) |
| ARP | $\frac{1}{K}\sum_{i \in R_u(K)} \lvert\mathcal{U}_i\rvert$ | average popularity of recommended items (popularity bias) |
| ILD | $\frac{1}{K(K-1)}\sum_{i \ne j \in R_u(K)} \left(1 - \frac{\lvert G_i \cap G_j\rvert}{\lvert G_i \cup G_j\rvert}\right)$, with $G_i$ the genres of item $i$ | how different the items within one list are |
| Exposure Gini | Gini coefficient of how often each item is recommended across all users | concentration of recommendations on few items |

Per-user values of every metric are stored for every run. Week 3 uses them for paired significance tests (Wilcoxon signed-rank) and for analysing performance by user activity.

---

## 7.4 Algorithms (Part 1)

### 7.4.1 MostPop (non-personalized baseline)
$$s(u,i) = \lvert\mathcal{U}_i\rvert$$
Every user receives the most-liked items they have not seen yet. Any personalized model must beat this to justify its complexity.

### 7.4.2 User-based k-nearest neighbours (UserKNN)
Two users are similar if they liked the same items. With a shrinkage term $\lambda$ that down-weights similarities supported by only a few common items:

$$\mathrm{sim}(u,v) = \frac{\lvert\mathcal{I}_u \cap \mathcal{I}_v\rvert}{\sqrt{\lvert\mathcal{I}_u\rvert}\sqrt{\lvert\mathcal{I}_v\rvert} + \lambda}, \qquad s(u,i) = \sum_{v \in N_k(u)} \mathrm{sim}(u,v)\, x_{vi}$$

where $N_k(u)$ is the set of the $k$ most similar other users. User–user similarities are computed batch by batch at prediction time, so memory stays proportional to (batch size × number of users).

### 7.4.3 Item-based k-nearest neighbours (ItemKNN; Sarwar et al., 2001)
Two items are similar if they were liked by the same users:

$$\mathrm{sim}(i,j) = \frac{\lvert\mathcal{U}_i \cap \mathcal{U}_j\rvert}{\sqrt{\lvert\mathcal{U}_i\rvert}\sqrt{\lvert\mathcal{U}_j\rvert} + \lambda}, \qquad s(u,i) = \sum_{j \in \mathcal{I}_u \cap N_k(i)} \mathrm{sim}(j,i)$$

Only the $k$ nearest neighbours of each item are kept, so the similarity matrix is sparse. Item similarities change slowly and can be precomputed. They also give natural explanations ("because you liked *j*").

### 7.4.4 EASE^R: Embarrassingly Shallow Autoencoder (Steck, 2019)
EASE learns an item–item weight matrix $B \in \mathbb{R}^{n \times n}$ that reconstructs each user's interactions from their *other* interactions:

$$\min_{B}\; \lVert X - XB \rVert_F^2 + \lambda \lVert B \rVert_F^2 \quad \text{subject to} \quad \mathrm{diag}(B) = 0$$

The constraint rules out the trivial solution $B = I$, where each item would predict itself. Adding the constraint with Lagrange multipliers and setting the gradient to zero gives a closed-form solution. With $P = (X^\top X + \lambda I)^{-1}$:

$$B_{ij} = -\frac{P_{ij}}{P_{jj}} \;\; (i \neq j), \qquad B_{jj} = 0, \qquad s(u,\cdot) = x_u B$$

Training is a single $n \times n$ matrix inversion ($O(n^3)$; about 2.5 s for 3,458 items). Scoring needs only the user's interaction vector $x_u$, so a brand-new user can be scored immediately without retraining, which the live application relies on.

### 7.4.5 BPR-MF: Matrix factorization with Bayesian Personalized Ranking (Rendle et al., 2009)
Each user and item gets a $d$-dimensional embedding, and the score is their inner product, $s(u,i) = \mathbf{p}_u^\top \mathbf{q}_i$. Instead of predicting ratings, BPR learns to **rank**. It samples triples $(u, i, j)$, where $u$ liked $i$ and $j$ is a uniformly random item $u$ has not interacted with, and maximizes the probability that $i$ is ranked above $j$:

$$\mathcal{L} = -\frac{1}{\lvert\mathcal{B}\rvert}\sum_{(u,i,j)\in\mathcal{B}} \ln \sigma\!\left(s(u,i) - s(u,j)\right) + \frac{\lambda}{2\lvert\mathcal{B}\rvert}\sum_{(u,i,j)\in\mathcal{B}}\left(\lVert\mathbf{p}_u\rVert^2 + \lVert\mathbf{q}_i\rVert^2 + \lVert\mathbf{q}_j\rVert^2\right)$$

where $\mathcal{B}$ is a mini-batch and $\sigma$ is the logistic function.
- **Negative sampling:** rejection sampling guarantees that $j$ is never one of $u$'s training items.
- **Optimization:** Adam.
- **Early stopping:** every 5 epochs the model is evaluated on validation NDCG@10. Training stops after 5 evaluations without improvement, and the best weights are restored. The best epoch count is reused when refitting on training + validation.

All later iterative models (7.4.6–7.4.10) share this early-stopping and refitting procedure.

### 7.4.6 iALS: Implicit Alternating Least Squares (Hu, Koren & Volinsky, 2008)
BPR samples a few negatives. iALS instead uses **every** user–item cell. A liked item is a positive with high confidence, and an unobserved item is a weak negative:

$$\min_{P,Q}\; \sum_{u,i} c_{ui}\left(x_{ui} - \mathbf{p}_u^\top\mathbf{q}_i\right)^2 + \lambda\left(\sum_u\lVert\mathbf{p}_u\rVert^2 + \sum_i\lVert\mathbf{q}_i\rVert^2\right), \qquad c_{ui} = 1 + \alpha\, x_{ui}$$

With $Q$ fixed, each user vector has an exact closed-form solution, and vice versa. Training alternates between the two:

$$\mathbf{p}_u = \left(Q^\top C^u Q + \lambda I\right)^{-1} Q^\top C^u \mathbf{x}_u, \qquad Q^\top C^u Q = Q^\top Q + \alpha\, Q_{\mathcal{I}_u}^\top Q_{\mathcal{I}_u}$$

The identity on the right is what makes iALS practical. $Q^\top Q$ is shared by all users, so each solve costs only $O(\lvert\mathcal{I}_u\rvert d^2 + d^3)$ instead of touching all $n$ items. The training objective is monitored without forming the dense $m \times n$ matrix: the sum over all cells of $(\mathbf{p}_u^\top\mathbf{q}_i)^2$ equals $\mathrm{tr}\!\left((P^\top P)(Q^\top Q)\right)$.

### 7.4.7 NeuMF: Neural Collaborative Filtering (He et al., 2017)
NeuMF replaces the fixed dot product with a learned interaction function made of two branches:

$$\boldsymbol{\phi}^{GMF} = \mathbf{p}_u^{G} \odot \mathbf{q}_i^{G}, \qquad \mathbf{z}_1 = \begin{bmatrix}\mathbf{p}_u^{M}\\ \mathbf{q}_i^{M}\end{bmatrix},\;\; \mathbf{z}_{l+1} = \mathrm{ReLU}(W_l\mathbf{z}_l + \mathbf{b}_l), \qquad \hat{y}_{ui} = \sigma\!\left(\mathbf{h}^\top\begin{bmatrix}\boldsymbol{\phi}^{GMF}\\ \mathbf{z}_L\end{bmatrix}\right)$$

The GMF branch is a weighted dot product. The MLP branch, with tower 64 → 32 → 16, can model non-linear interactions. Training is binary classification with the cross-entropy loss: each liked pair is a positive, plus 4 randomly sampled unobserved items per positive as negatives.

**Cost note:** to rank the full catalog, the MLP must run on every (user, item) pair ($m \times n$ forward passes). Dot-product models need one matrix multiplication instead. This is a practical disadvantage of NeuMF that Rendle et al. (2020) also point out.

### 7.4.8 Mult-VAE: Variational Autoencoder for CF (Liang et al., 2018)
Mult-VAE is a generative model of a user's whole interaction vector. A latent taste vector $\mathbf{z}_u \sim \mathcal{N}(0, I)$ is decoded into a probability distribution over all items, and the user's interactions are drawn from it:

$$\boldsymbol{\pi}(\mathbf{z}_u) = \mathrm{softmax}\!\left(f_\theta(\mathbf{z}_u)\right), \qquad \mathbf{x}_u \sim \mathrm{Multinomial}\!\left(N_u, \boldsymbol{\pi}(\mathbf{z}_u)\right)$$

An encoder $q_\phi(\mathbf{z}_u \mid \mathbf{x}_u) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}_u), \mathrm{diag}\,\boldsymbol{\sigma}^2_\phi(\mathbf{x}_u))$ approximates the posterior. Training maximizes the β-weighted evidence lower bound:

$$\mathcal{L}_\beta(\mathbf{x}_u) = \sum_{i} x_{ui}\log \pi_i(\mathbf{z}_u) \;-\; \beta\,\mathrm{KL}\!\left(q_\phi(\mathbf{z}_u\mid\mathbf{x}_u)\,\Vert\,\mathcal{N}(0, I)\right), \qquad \mathbf{z}_u = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$$

- **β annealing:** β rises linearly from 0 to 0.2 over 100 epochs. Liang et al. found that partial regularization (β < 1) works best.
- **Architecture:** $n$ → 600 → 200 → 600 → $n$ with tanh activations; the input is L2-normalized, with dropout 0.5.
- **Prediction:** $\mathbf{z}_u = \boldsymbol{\mu}$ (no sampling). The decoder logits rank the items.

Like EASE, Mult-VAE scores a user from their interaction vector alone, so new users need no retraining.

### 7.4.9 LightGCN: Light Graph Convolution Network (He et al., 2020)
Users and items are nodes of one bipartite graph $A = \begin{bmatrix}0 & X\\ X^\top & 0\end{bmatrix}$. Each layer replaces every embedding with the normalized sum of its neighbours' embeddings, and the final embedding averages all layers:

$$\hat{A} = D^{-1/2} A D^{-1/2}, \qquad E^{(k+1)} = \hat{A} E^{(k)}, \qquad E = \frac{1}{K+1}\sum_{k=0}^{K} E^{(k)}, \qquad s(u,i) = \mathbf{e}_u^\top\mathbf{e}_i$$

- **Multi-hop signal:** after two layers, a user's embedding contains items liked by users with similar taste. Matrix factorization never sees these indirect connections.
- **What LightGCN removed:** its predecessor NGCF used weight matrices and non-linear activations in every layer. He et al. showed these hurt accuracy, and kept only the propagation.
- **Training:** the BPR loss of Section 7.4.5, with L2 regularization on the layer-0 embeddings $E^{(0)}$; $K = 3$ layers.

**Engineering for CPU training.** Every training step must propagate the whole graph (9,496 nodes, 919,202 edges) forward and backward. With PyTorch's generic sparse backward, the three layers took 647 ms per step. Because $\hat{A}$ is symmetric, the gradient of $E^{(k+1)} = \hat{A}E^{(k)}$ is simply

$$\frac{\partial\mathcal{L}}{\partial E^{(k)}} = \hat{A}^\top\frac{\partial\mathcal{L}}{\partial E^{(k+1)}} = \hat{A}\,\frac{\partial\mathcal{L}}{\partial E^{(k+1)}}$$

so we implemented the backward pass as one more multiplication by the same sparse matrix: 58 ms, an 11× speed-up. A unit test checks that this gradient matches PyTorch's exact gradient. We also used batches of 8,192 with learning rate 0.002 (the paper uses 2,048 and 0.001), which means 4× fewer propagations per epoch. Together, one epoch fell from 141 s to 6.5 s, so LightGCN trains on a laptop without a GPU.

### 7.4.10 SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
SASRec reads a user's history as an ordered sequence, like a sentence, and predicts the next item. The input is the last $L = 200$ items $(s_1, \dots, s_n)$ in time order:

$$\mathbf{h}^{(0)}_t = \sqrt{d}\,\mathbf{m}_{s_t} + \mathbf{p}_t, \qquad \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d} + M\right)V, \qquad M_{ts} = \begin{cases}-\infty & s > t \text{ (future) or } s \text{ is padding}\\ 0 & \text{otherwise}\end{cases}$$

- **Architecture:** two blocks, each with causal self-attention and a point-wise feed-forward network, using residual connections, layer normalization and dropout 0.2.
- **Causality:** the mask $M$ makes the hidden state $\mathbf{F}_t$ at position $t$ depend only on items up to $t$.
- **Prediction:** the score of item $i$ as the next item is $r_{i,t} = \mathbf{F}_t^\top\mathbf{m}_i$, which reuses the input item embeddings.

Two training objectives are compared:

$$\mathcal{L}_{BCE} = -\sum_t\left[\log\sigma(r_{s_{t+1},t}) + \log\left(1-\sigma(r_{j,t})\right)\right] \;\;\text{(paper; one sampled negative } j\text{)}, \qquad \mathcal{L}_{CE} = -\sum_t \log\frac{\exp r_{s_{t+1},t}}{\sum_{i}\exp r_{i,t}}$$

The full-softmax cross-entropy variant was later shown to be much stronger (Klenitskiy & Vasilev, 2023). With 3,458 items it is cheap to compute.

**Engineering details.** Three implementation issues were found and fixed during development:
1. **Initialization.** With PyTorch's default embeddings ($\mathcal{N}(0,1)$), the initial scores were about ±64, the first-epoch BCE loss averaged 5.5 although it should start near $2\ln 2 \approx 1.39$, and the model failed to learn (validation NDCG 0.005). Small Xavier-normal weights, as in the original code, fixed this. A unit test now checks the starting loss.
2. **Padding invariance.** Padding positions are excluded from attention, and position indices are counted from the most recent item. As a result, a sequence gets the same hidden states regardless of how much padding its batch has. A unit test checks this.
3. **Length-bucketed batches.** Thanks to point 2, users with similar history lengths can be batched together and each batch trimmed to its longest sequence. Attention cost grows with $L^2$, so a training step fell from 1,010 ms at length 200 to 157 ms at a typical length of 46.

**A known limitation on this dataset:** 77% of MovieLens-1M ratings share their exact second with another rating by the same user (Chapter 6). The order inside such a burst is therefore set by our seeded tie-break rather than by the user, so part of the "sequence" SASRec learns from is artificial.

### 7.4.11 Rating prediction with biased matrix factorization (experiment E7)
For completeness and comparison with older literature, we also run the Netflix-Prize task of predicting the star rating itself. Here every rating counts, not only ratings ≥ 4, with the same chronological protocol. The "SVD" model of Koren, Bell & Volinsky (2009) is:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top\mathbf{q}_i, \qquad \min \sum_{(u,i)}\left(r_{ui} - \hat{r}_{ui}\right)^2 + \lambda\left(b_u^2 + b_i^2 + \lVert\mathbf{p}_u\rVert^2 + \lVert\mathbf{q}_i\rVert^2\right)$$

It is compared against the global mean $\mu$ and against a bias-only model $\mu + b_u + b_i$. The bias-only model uses shrinkage, for example $b_i = \sum_{u}(r_{ui}-\mu-b_u)/(\lambda_i + \lvert R(i)\rvert)$, so an item rated by few users is pulled towards the average. Accuracy is measured with

$$\mathrm{RMSE} = \sqrt{\tfrac{1}{\lvert T\rvert}\textstyle\sum_{(u,i)\in T}(r_{ui}-\hat{r}_{ui})^2}, \qquad \mathrm{MAE} = \tfrac{1}{\lvert T\rvert}\textstyle\sum_{(u,i)\in T}\lvert r_{ui}-\hat{r}_{ui}\rvert$$

with predictions clipped to the 1–5 star range.

---

## 7.5 Implementation and reproducibility

| Item | Choice |
|---|---|
| Language and libraries | Python 3.12; NumPy, SciPy (sparse matrices), pandas, PyTorch 2.15 (CPU) |
| Hardware | Laptop CPU with 8 logical cores, no GPU |
| Experiment tracking | MLflow: parameters, all metrics, training curves, per-user results, split fingerprint |
| Randomness | fixed seeds for the split (tie-break) and for model training |
| Configuration | one YAML file per dataset (`ml/configs/ml-1m.yaml`) |

**Week 1 hyperparameters** (literature defaults, not yet tuned; Week 3 tunes every model with the same Optuna budget):

| Model | Hyperparameters |
|---|---|
| UserKNN, ItemKNN | $k = 100$ neighbours, shrinkage $\lambda = 10$ |
| EASE | $\lambda = 500$ |
| BPR-MF | $d = 64$, learning rate $10^{-3}$, $\lambda = 10^{-4}$, batch size 4,096, at most 300 epochs, patience 5 evaluations |
| iALS | $d = 64$, $\alpha = 10$, $\lambda = 1$, at most 30 sweeps, validated after every sweep, patience 3 |
| NeuMF | GMF $d = 32$, MLP tower 64 → 32 → 16, 4 negatives per positive, learning rate $10^{-3}$, batch 2,048, at most 50 epochs |
| Mult-VAE | 600 → 200 → 600, dropout 0.5, β annealed to 0.2 over 100 epochs, learning rate $10^{-3}$, batch 500 users |
| LightGCN | $d = 64$, $K = 3$ layers, $\lambda = 10^{-4}$, learning rate $2 \times 10^{-3}$, batch 8,192 (paper: 2,048 and $10^{-3}$; changed for CPU training) |
| SASRec | $d = 64$ (paper: 50), 2 blocks, 1 head, $L = 200$, dropout 0.2, learning rate $10^{-3}$, batch 128, at most 200 epochs; BCE loss (paper) and CE loss variant |
| Biased MF (rating) | $d = 64$, $\lambda = 0.02$, learning rate $10^{-3}$, batch 8,192, early stopping on validation RMSE |

---

## 7.6 Verification of the methodology

The evaluation code is the measuring instrument of the whole study, so it is tested before any model is compared (55 automated tests):

- **Metrics:** each metric is compared with values worked out by hand for small examples. For example, a user with targets at ranks 2 and 5 out of 2 targets has NDCG@5 = 0.6241 and AP@5 = 0.45.
- **Evaluator:** items in the user's history are never recommended, users without targets are skipped, the model's own scores are never modified, and invalid scores (NaN, wrong shape) are rejected.
- **Split:** no user–item pair appears in two parts, no validation or test interaction is older than a training interaction of the same user, the result does not depend on input row order, and tampered split files are detected through the fingerprint.
- **Models:**
  - KNN scores match hand-computed cosine examples.
  - EASE's closed form equals solving a separate ridge regression for every item, which is exactly what the zero-diagonal constraint means.
  - BPR learns a planted two-group taste structure (AUC > 0.95), and its negative sampler never returns a positive item.
- **Deep and graph models (Week 2):**
  - iALS's fast per-user solve equals the textbook weighted least-squares solution with an explicit confidence matrix, and its training objective matches a brute-force computation.
  - LightGCN's normalized adjacency matches hand-computed values, and its custom symmetric backward pass gives exactly PyTorch's gradient.
  - SASRec never uses future items (changing later items leaves earlier hidden states unchanged), is invariant to padding, starts training at the theoretical loss $2\ln 2$, and learns a planted "next item" pattern with both losses.
  - NeuMF, Mult-VAE, LightGCN and iALS each learn the planted two-group structure. NeuMF's chunked scoring is exact, and Mult-VAE's predictions are deterministic.
  - Rating prediction: RMSE/MAE match hand computations, the bias model recovers exactly additive ratings, and biased MF beats the bias model on low-rank data.
