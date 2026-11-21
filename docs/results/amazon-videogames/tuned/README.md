# Results: amazon-videogames (tuned)

Dataset **amazon-videogames**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `39a61d28e43ea969`): 42,045 users, 5,768 items; train/val/test interactions = 251,137 / 47,187 / 47,148.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `amazon-videogames-tuned.yaml`. Tuned on the validation set with Optuna (TPE), equal budget per model: at most 30 trials or 10 minutes, trial 0 = paper defaults. Test data was never used for tuning.

Every number comes from the latest MLflow run with exactly these settings on this split.

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0215 | 0.0435 | 0.0470 | 0.0047 | 0.0157 | 0.0145 | 0.0270 | 0.0648 | 0.0031 | 5.3628 | n/a | 1042.0 | 0.0 |
| UserKNN | Neighbourhood | 0.0629 | 0.1116 | 0.1174 | 0.0119 | 0.0495 | 0.0471 | 0.0745 | 0.1570 | 0.4977 | 6.9640 | n/a | 484.4 | 0.0 |
| ItemKNN | Neighbourhood | 0.0647 | 0.1139 | 0.1195 | 0.0121 | 0.0510 | 0.0487 | 0.0769 | 0.1617 | 0.7816 | 7.7771 | n/a | 302.8 | 1.4 |
| EASE | Linear autoencoder | 0.0706 | 0.1221 | 0.1286 | 0.0130 | 0.0565 | 0.0538 | 0.0825 | 0.1683 | 0.7741 | 7.7090 | n/a | 324.0 | 17.7 |
| Markov chain | Sequential (Markov) | 0.0618 | 0.1002 | 0.1075 | 0.0109 | 0.0522 | 0.0488 | 0.0711 | 0.1364 | 0.8115 | 7.5883 | n/a | 357.8 | 0.0 |
| BPR-MF | Latent factor | 0.0321 | 0.0579 | 0.0610 | 0.0062 | 0.0250 | 0.0239 | 0.0392 | 0.0856 | **0.9816** | **8.4273** | n/a | 201.3 | 924.3 |
| iALS | Latent factor | 0.0585 | 0.1049 | 0.1103 | 0.0112 | 0.0459 | 0.0436 | 0.0706 | 0.1520 | 0.2424 | 7.4863 | n/a | 300.7 | 111.3 |
| LightGCN | Graph neural network | 0.0586 | 0.1053 | 0.1107 | 0.0112 | 0.0456 | 0.0436 | 0.0707 | 0.1527 | 0.7842 | 7.7032 | n/a | 311.7 | 1536.2 |
| SASRec (CE loss) | Sequential (Transformer) | 0.0849 | 0.1453 | 0.1547 | 0.0158 | 0.0690 | 0.0649 | 0.0997 | 0.2026 | 0.7042 | 7.8603 | n/a | 309.6 | 318.5 |
| Red Box hybrid | Two-stage hybrid (ours) | **0.0919** | **0.1552** | **0.1653** | **0.0169** | **0.0756** | **0.0709** | **0.1077** | **0.2168** | 0.7557 | 7.6591 | n/a | 351.0 | 153.5 |

## Validation set

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0198 | 0.0400 | 0.0430 | 0.0043 | 0.0144 | 0.0134 | 0.0261 | 0.0644 | 0.0029 | 5.6430 | n/a | 860.4 | 0.0 |
| UserKNN | Neighbourhood | 0.0700 | 0.1244 | 0.1308 | 0.0132 | 0.0550 | 0.0524 | 0.0832 | 0.1763 | 0.5945 | 7.3192 | n/a | 383.4 | 0.0 |
| ItemKNN | Neighbourhood | 0.0718 | 0.1270 | 0.1335 | 0.0135 | 0.0566 | 0.0540 | 0.0855 | 0.1802 | 0.8613 | 8.1212 | n/a | 239.0 | 1.5 |
| EASE | Linear autoencoder | 0.0758 | 0.1317 | 0.1391 | 0.0141 | 0.0605 | 0.0575 | 0.0885 | 0.1812 | 0.8282 | 7.9970 | n/a | 263.7 | 23.2 |
| Markov chain | Sequential (Markov) | 0.0635 | 0.1044 | 0.1119 | 0.0113 | 0.0533 | 0.0497 | 0.0725 | 0.1396 | 0.8620 | 7.8336 | n/a | 300.1 | 0.0 |
| BPR-MF | Latent factor | 0.0345 | 0.0625 | 0.0669 | 0.0067 | 0.0269 | 0.0254 | 0.0422 | 0.0925 | **0.9667** | **8.6526** | n/a | 169.2 | 1218.8 |
| iALS | Latent factor | 0.0633 | 0.1160 | 0.1226 | 0.0124 | 0.0488 | 0.0464 | 0.0770 | 0.1692 | 0.2387 | 7.7319 | n/a | 251.2 | 341.3 |
| LightGCN | Graph neural network | 0.0639 | 0.1169 | 0.1232 | 0.0125 | 0.0493 | 0.0469 | 0.0781 | 0.1723 | 0.7807 | 7.9816 | n/a | 253.5 | 1822.2 |
| SASRec (CE loss) | Sequential (Transformer) | 0.0850 | 0.1484 | 0.1583 | 0.0162 | 0.0684 | 0.0639 | 0.1006 | 0.2092 | 0.7259 | 8.1014 | n/a | 255.7 | 804.7 |
| Red Box hybrid | Two-stage hybrid (ours) | **0.2061** | **0.3602** | **0.3835** | **0.0391** | **0.1670** | **0.1553** | **0.2407** | **0.4952** | 0.4707 | 7.9407 | n/a | 283.0 | 171.1 |
