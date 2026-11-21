# Week 1 baselines: ml-1m

Dataset **ml-1m**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `8d6a7a5a8a4c83ab`): 6,038 users, 3,458 items; train/val/test interactions = 459,601 / 57,805 / 57,773.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `ml-1m.yaml` (literature defaults, **not yet tuned**; equal-budget Optuna tuning follows in Week 3).

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | 0.0466 | 0.0445 | 0.2322 | 0.0350 | 0.0893 | 0.0213 | 0.0570 | 0.0788 | 0.0298 | 1.5664 | **0.7750** | 2061.0 | 0.0 |
| UserKNN | 0.0727 | 0.0772 | 0.3182 | 0.0477 | 0.1272 | 0.0366 | 0.0906 | 0.1352 | 0.1342 | 2.0974 | 0.7536 | 1542.0 | 0.0 |
| ItemKNN | 0.0709 | 0.0790 | 0.3270 | 0.0474 | 0.1223 | 0.0345 | 0.0888 | 0.1346 | 0.2869 | 2.4784 | 0.7140 | 1270.6 | 1.0 |
| EASE | **0.0810** | **0.0936** | **0.3629** | **0.0519** | **0.1368** | **0.0399** | **0.1025** | **0.1604** | 0.2331 | 2.5026 | 0.7480 | 1234.1 | 2.5 |
| BPR-MF | 0.0685 | 0.0751 | 0.3230 | 0.0459 | 0.1224 | 0.0327 | 0.0860 | 0.1296 | **0.4130** | **2.7305** | 0.7155 | 1147.6 | 125.8 |

## Validation set

| model | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | 0.0507 | 0.0469 | 0.2401 | 0.0380 | 0.0962 | 0.0238 | 0.0637 | 0.0901 | 0.0249 | 1.6417 | **0.7608** | 1956.3 | 0.0 |
| UserKNN | 0.0792 | 0.0870 | 0.3387 | 0.0516 | 0.1335 | 0.0401 | 0.1000 | 0.1521 | 0.1313 | 2.1963 | 0.7487 | 1449.8 | 0.0 |
| ItemKNN | 0.0765 | 0.0877 | 0.3386 | 0.0506 | 0.1265 | 0.0377 | 0.0978 | 0.1521 | 0.2811 | 2.5958 | 0.7076 | 1185.0 | 0.9 |
| EASE | **0.0867** | **0.1019** | **0.3805** | **0.0548** | **0.1418** | **0.0434** | **0.1091** | **0.1711** | 0.2290 | 2.5910 | 0.7450 | 1169.0 | 2.3 |
| BPR-MF | 0.0709 | 0.0792 | 0.3286 | 0.0469 | 0.1249 | 0.0340 | 0.0899 | 0.1367 | **0.3892** | **2.7834** | 0.7133 | 1114.2 | 130.9 |
