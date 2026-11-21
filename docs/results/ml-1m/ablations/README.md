# Results: ml-1m (ablations)

Dataset **ml-1m**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `8d6a7a5a8a4c83ab`): 6,038 users, 3,458 items; train/val/test interactions = 459,601 / 57,805 / 57,773.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `ml-1m-ablations.yaml`. Ablations. Each entry changes exactly one thing relative to its reference model in ml-1m.yaml.

Every number comes from the latest MLflow run with exactly these settings on this split.

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SASRec (CE, shuffled order) | Sequential (Transformer) | **0.0633** | **0.0717** | **0.3000** | **0.0425** | **0.1114** | **0.0299** | **0.0811** | **0.1269** | **0.3973** | **2.9445** | **0.6868** | 1026.5 | 1435.1 |

## Validation set

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SASRec (CE, shuffled order) | Sequential (Transformer) | **0.0685** | **0.0801** | **0.3180** | **0.0448** | **0.1152** | **0.0331** | **0.0895** | **0.1433** | **0.3791** | **3.0167** | **0.6898** | 978.0 | 2130.0 |
