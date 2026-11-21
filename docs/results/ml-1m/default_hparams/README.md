# Results: ml-1m (default_hparams)

Dataset **ml-1m**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `8d6a7a5a8a4c83ab`): 6,038 users, 3,458 items; train/val/test interactions = 459,601 / 57,805 / 57,773.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `ml-1m.yaml`. Literature-default hyperparameters, **not yet tuned** (equal-budget Optuna tuning follows in Week 3). Deviations from the papers, all made to fit CPU training, are noted in ml/configs/ml-1m.yaml.

Every number comes from the latest MLflow run with exactly these settings on this split.

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0466 | 0.0445 | 0.2322 | 0.0350 | 0.0893 | 0.0213 | 0.0570 | 0.0788 | 0.0298 | 1.5664 | **0.7750** | 2061.0 | 0.0 |
| UserKNN | Neighbourhood | 0.0727 | 0.0772 | 0.3182 | 0.0477 | 0.1272 | 0.0366 | 0.0906 | 0.1352 | 0.1342 | 2.0974 | 0.7536 | 1542.0 | 0.0 |
| ItemKNN | Neighbourhood | 0.0709 | 0.0790 | 0.3270 | 0.0474 | 0.1223 | 0.0345 | 0.0888 | 0.1346 | 0.2869 | 2.4784 | 0.7140 | 1270.6 | 1.0 |
| EASE | Linear autoencoder | 0.0810 | 0.0936 | 0.3629 | 0.0519 | 0.1368 | 0.0399 | 0.1025 | 0.1604 | 0.2331 | 2.5026 | 0.7480 | 1234.1 | 2.5 |
| BPR-MF | Latent factor | 0.0685 | 0.0751 | 0.3230 | 0.0459 | 0.1224 | 0.0327 | 0.0860 | 0.1296 | 0.4130 | 2.7305 | 0.7155 | 1147.6 | 125.8 |
| Markov chain | Sequential (Markov) | 0.1097 | 0.1170 | 0.4156 | 0.0638 | 0.1896 | 0.0605 | 0.1248 | 0.1683 | 0.5014 | 3.2965 | 0.6796 | 823.0 | 0.0 |
| iALS | Latent factor | 0.0707 | 0.0833 | 0.3300 | 0.0443 | 0.1232 | 0.0341 | 0.0904 | 0.1445 | 0.3768 | 3.0034 | 0.7181 | 952.7 | 79.5 |
| NeuMF | Deep (MLP) | 0.0683 | 0.0760 | 0.3199 | 0.0454 | 0.1201 | 0.0326 | 0.0878 | 0.1368 | 0.3728 | 2.8990 | 0.6975 | 1035.0 | 334.0 |
| Mult-VAE | Deep (VAE) | 0.0697 | 0.0793 | 0.3227 | 0.0449 | 0.1205 | 0.0341 | 0.0911 | 0.1447 | 0.4488 | 2.9005 | 0.7098 | 1053.7 | 474.4 |
| LightGCN | Graph neural network | 0.0736 | 0.0838 | 0.3445 | 0.0485 | 0.1280 | 0.0350 | 0.0935 | 0.1466 | 0.4728 | 2.9646 | 0.7104 | 1028.1 | 2684.0 |
| SASRec | Sequential (Transformer) | 0.1382 | 0.1511 | 0.5204 | 0.0878 | 0.2305 | 0.0735 | 0.1647 | 0.2373 | 0.5538 | 3.7383 | 0.6314 | 645.8 | 1638.4 |
| SASRec (CE loss) | Sequential (Transformer) | **0.1643** | **0.1796** | **0.5621** | **0.1022** | **0.2606** | **0.0919** | **0.1927** | **0.2710** | **0.6400** | **3.9973** | 0.6022 | 569.3 | 3775.6 |

## Validation set

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0507 | 0.0469 | 0.2401 | 0.0380 | 0.0962 | 0.0238 | 0.0637 | 0.0901 | 0.0249 | 1.6417 | **0.7608** | 1956.3 | 0.0 |
| UserKNN | Neighbourhood | 0.0792 | 0.0870 | 0.3387 | 0.0516 | 0.1335 | 0.0401 | 0.1000 | 0.1521 | 0.1313 | 2.1963 | 0.7487 | 1449.8 | 0.0 |
| ItemKNN | Neighbourhood | 0.0765 | 0.0877 | 0.3386 | 0.0506 | 0.1265 | 0.0377 | 0.0978 | 0.1521 | 0.2811 | 2.5958 | 0.7076 | 1185.0 | 0.9 |
| EASE | Linear autoencoder | 0.0867 | 0.1019 | 0.3805 | 0.0548 | 0.1418 | 0.0434 | 0.1091 | 0.1711 | 0.2290 | 2.5910 | 0.7450 | 1169.0 | 2.3 |
| BPR-MF | Latent factor | 0.0709 | 0.0792 | 0.3286 | 0.0469 | 0.1249 | 0.0340 | 0.0899 | 0.1367 | 0.3892 | 2.7834 | 0.7133 | 1114.2 | 130.9 |
| Markov chain | Sequential (Markov) | 0.1183 | 0.1257 | 0.4402 | 0.0692 | 0.2045 | 0.0654 | 0.1363 | 0.1858 | 0.5035 | 3.3622 | 0.6830 | 793.6 | 0.0 |
| iALS | Latent factor | 0.0730 | 0.0917 | 0.3379 | 0.0447 | 0.1177 | 0.0359 | 0.0951 | 0.1568 | 0.3580 | 3.1009 | 0.7178 | 898.6 | 46.2 |
| NeuMF | Deep (MLP) | 0.0716 | 0.0841 | 0.3346 | 0.0469 | 0.1222 | 0.0343 | 0.0921 | 0.1447 | 0.3632 | 2.9433 | 0.6964 | 1009.8 | 290.6 |
| Mult-VAE | Deep (VAE) | 0.0757 | 0.0929 | 0.3462 | 0.0484 | 0.1232 | 0.0367 | 0.0970 | 0.1564 | 0.4291 | 2.9810 | 0.7104 | 1003.4 | 826.1 |
| LightGCN | Graph neural network | 0.0771 | 0.0912 | 0.3530 | 0.0492 | 0.1301 | 0.0374 | 0.0984 | 0.1555 | 0.4505 | 3.0528 | 0.7073 | 974.5 | 2440.3 |
| SASRec | Sequential (Transformer) | 0.1506 | 0.1655 | 0.5510 | 0.0956 | 0.2466 | 0.0808 | 0.1808 | 0.2623 | 0.5330 | 3.7169 | 0.6273 | 645.5 | 2563.0 |
| SASRec (CE loss) | Sequential (Transformer) | **0.1784** | **0.1937** | **0.5892** | **0.1099** | **0.2831** | **0.1010** | **0.2108** | **0.2977** | **0.6056** | **3.9569** | 0.6019 | 574.6 | 6178.7 |
