# Results: ml-32m (default_hparams)

Dataset **ml-32m**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `886c0f151b9f83b6`): 42,144 users, 8,487 items; train/val/test interactions = 2,820,227 / 355,408 / 355,408.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `ml-32m.yaml`. App models on MovieLens-32M (2018-2023 window). The same protocol as the research benchmark: per-user chronological 80/10/10 split, full-catalogue ranking, test set used once.

Every number comes from the latest MLflow run with exactly these settings on this split.

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0374 | 0.0403 | 0.1815 | 0.0247 | 0.0693 | 0.0178 | 0.0486 | 0.0740 | 0.0157 | 1.6622 | **0.7841** | 13599.7 | 0.0 |
| Markov chain | Sequential (Markov) | 0.0745 | 0.0818 | 0.2963 | 0.0405 | 0.1346 | 0.0406 | 0.0876 | 0.1239 | 0.5916 | 3.9558 | 0.7513 | 4486.3 | 0.3 |
| EASE | Linear autoencoder | 0.0911 | 0.1016 | 0.3818 | 0.0561 | 0.1596 | 0.0466 | 0.1117 | 0.1656 | 0.2504 | 2.8807 | 0.7515 | 7240.5 | 43.4 |
| iALS | Latent factor | 0.0804 | 0.0915 | 0.3523 | 0.0499 | 0.1411 | 0.0402 | 0.1004 | 0.1518 | 0.1747 | 2.9810 | 0.7413 | 6579.6 | 182.8 |
| SASRec (CE loss) | Sequential (Transformer) | **0.1125** | **0.1220** | **0.4306** | **0.0661** | **0.1962** | **0.0609** | **0.1336** | **0.1891** | **0.6326** | **4.3871** | 0.6935 | 3643.0 | 5327.6 |

## Validation set

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0399 | 0.0444 | 0.1907 | 0.0262 | 0.0715 | 0.0191 | 0.0513 | 0.0790 | 0.0143 | 1.7168 | **0.7834** | 13114.3 | 0.0 |
| Markov chain | Sequential (Markov) | 0.0815 | 0.0901 | 0.3143 | 0.0437 | 0.1436 | 0.0454 | 0.0963 | 0.1367 | 0.5878 | 4.0018 | 0.7521 | 4358.5 | 0.3 |
| EASE | Linear autoencoder | 0.0954 | 0.1100 | 0.3934 | 0.0580 | 0.1607 | 0.0494 | 0.1184 | 0.1796 | 0.2395 | 2.9480 | 0.7501 | 6953.8 | 38.4 |
| iALS | Latent factor | 0.0859 | 0.1002 | 0.3667 | 0.0527 | 0.1458 | 0.0437 | 0.1081 | 0.1662 | 0.1608 | 3.0751 | 0.7413 | 6193.1 | 805.0 |
| SASRec (CE loss) | Sequential (Transformer) | **0.1218** | **0.1357** | **0.4525** | **0.0705** | **0.2045** | **0.0670** | **0.1449** | **0.2069** | **0.6060** | **4.4095** | 0.6945 | 3595.6 | 7464.9 |
