# Results: ml-1m (tuned)

Dataset **ml-1m**, split `pos4_val0.1_test0.1_min3_seed42` (fingerprint `8d6a7a5a8a4c83ab`): 6,038 users, 3,458 items; train/val/test interactions = 459,601 / 57,805 / 57,773.

Protocol: fit on train (iterative models early-stop on validation ndcg@10), then refit on train+validation and evaluate once on test.

Hyperparameters: `ml-1m-tuned.yaml`. Tuned on the validation set with Optuna (TPE), equal budget per model: at most 40 trials or 25 minutes, trial 0 = paper defaults. Test data was never used for tuning.

Every number comes from the latest MLflow run with exactly these settings on this split.

All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). Higher is better for every column except ARP@10 (average popularity of recommended items: lower means less popularity bias, which is not automatically better). Best value per column in **bold**. fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at prediction time instead (logged in MLflow as eval seconds).

## Test set (final numbers)

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0466 | 0.0445 | 0.2322 | 0.0350 | 0.0893 | 0.0213 | 0.0570 | 0.0788 | 0.0298 | 1.5664 | 0.7750 | 2061.0 | 0.0 |
| UserKNN | Neighbourhood | 0.0724 | 0.0785 | 0.3222 | 0.0480 | 0.1253 | 0.0360 | 0.0898 | 0.1346 | 0.1683 | 2.1835 | 0.7551 | 1474.7 | 0.0 |
| ItemKNN | Neighbourhood | 0.0696 | 0.0734 | 0.3099 | 0.0467 | 0.1222 | 0.0346 | 0.0847 | 0.1224 | 0.2050 | 2.1794 | 0.7433 | 1479.3 | 0.8 |
| EASE | Linear autoencoder | 0.0810 | 0.0935 | 0.3629 | 0.0520 | 0.1369 | 0.0398 | 0.1025 | 0.1604 | 0.2322 | 2.4997 | 0.7481 | 1235.9 | 1.8 |
| BPR-MF | Latent factor | 0.0740 | 0.0840 | 0.3394 | 0.0488 | 0.1289 | 0.0354 | 0.0939 | 0.1458 | 0.4138 | 2.8366 | 0.7112 | 1082.9 | 420.0 |
| Markov chain | Sequential (Markov) | 0.1097 | 0.1170 | 0.4156 | 0.0638 | 0.1896 | 0.0605 | 0.1248 | 0.1683 | 0.5014 | 3.2965 | 0.6796 | 823.0 | 0.0 |
| iALS | Latent factor | 0.0726 | 0.0854 | 0.3384 | 0.0466 | 0.1249 | 0.0348 | 0.0920 | 0.1470 | 0.2915 | 2.8524 | 0.7256 | 1008.5 | 6.4 |
| NeuMF | Deep (MLP) | 0.0698 | 0.0786 | 0.3291 | 0.0459 | 0.1234 | 0.0331 | 0.0888 | 0.1377 | 0.4372 | 2.9257 | 0.7124 | 1034.7 | 307.0 |
| Mult-VAE | Deep (VAE) | 0.0697 | 0.0793 | 0.3227 | 0.0449 | 0.1205 | 0.0341 | 0.0911 | 0.1447 | 0.4488 | 2.9005 | 0.7098 | 1053.7 | 474.4 |
| LightGCN | Graph neural network | 0.0736 | 0.0838 | 0.3445 | 0.0485 | 0.1280 | 0.0350 | 0.0935 | 0.1466 | 0.4728 | 2.9646 | 0.7104 | 1028.1 | 2684.0 |
| SASRec | Sequential (Transformer) | 0.1382 | 0.1511 | 0.5204 | 0.0878 | 0.2305 | 0.0735 | 0.1647 | 0.2373 | 0.5538 | 3.7383 | 0.6314 | 645.8 | 1638.4 |
| SASRec (CE loss) | Sequential (Transformer) | 0.1643 | 0.1796 | 0.5621 | 0.1022 | 0.2606 | 0.0919 | 0.1927 | 0.2710 | **0.6400** | **3.9973** | 0.6022 | 569.3 | 3775.6 |
| Red Box hybrid | Two-stage hybrid (ours) | **0.1790** | **0.1920** | **0.5966** | **0.1102** | **0.2873** | **0.1016** | **0.2072** | **0.2848** | 0.5315 | 3.5246 | 0.6350 | 718.8 | 44.7 |
| hybrid_mmr0.9 | Two-stage hybrid (ours) | 0.1768 | 0.1894 | 0.5948 | 0.1084 | 0.2869 | 0.0998 | 0.2064 | 0.2842 | 0.5220 | 3.4808 | 0.6862 | 737.5 | 44.7 |
| hybrid_mmr0.8 | Two-stage hybrid (ours) | 0.1716 | 0.1831 | 0.5872 | 0.1051 | 0.2837 | 0.0960 | 0.2022 | 0.2805 | 0.5110 | 3.4403 | 0.7376 | 756.9 | 44.7 |
| hybrid_mmr0.7 | Two-stage hybrid (ours) | 0.1653 | 0.1757 | 0.5807 | 0.1000 | 0.2802 | 0.0915 | 0.1955 | 0.2720 | 0.4997 | 3.3928 | 0.7843 | 782.3 | 44.7 |
| hybrid_mmr0.5 | Two-stage hybrid (ours) | 0.1467 | 0.1525 | 0.5441 | 0.0854 | 0.2680 | 0.0797 | 0.1710 | 0.2324 | 0.4803 | 3.2964 | **0.8582** | 837.0 | 44.7 |

## Validation set

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MostPop | Baseline | 0.0507 | 0.0469 | 0.2401 | 0.0380 | 0.0962 | 0.0238 | 0.0637 | 0.0901 | 0.0249 | 1.6417 | 0.7608 | 1956.3 | 0.0 |
| UserKNN | Neighbourhood | 0.0810 | 0.0890 | 0.3424 | 0.0518 | 0.1371 | 0.0414 | 0.1008 | 0.1506 | 0.1645 | 2.2848 | 0.7491 | 1385.6 | 0.0 |
| ItemKNN | Neighbourhood | 0.0787 | 0.0838 | 0.3313 | 0.0515 | 0.1332 | 0.0406 | 0.0967 | 0.1404 | 0.2079 | 2.2816 | 0.7372 | 1388.4 | 0.7 |
| EASE | Linear autoencoder | 0.0868 | 0.1020 | 0.3808 | 0.0547 | 0.1421 | 0.0435 | 0.1092 | 0.1710 | 0.2279 | 2.5882 | 0.7448 | 1170.7 | 1.9 |
| BPR-MF | Latent factor | 0.0801 | 0.0937 | 0.3610 | 0.0516 | 0.1324 | 0.0396 | 0.1010 | 0.1573 | 0.4023 | 2.9170 | 0.7102 | 1035.8 | 447.1 |
| Markov chain | Sequential (Markov) | 0.1183 | 0.1257 | 0.4402 | 0.0692 | 0.2045 | 0.0654 | 0.1363 | 0.1858 | 0.5035 | 3.3622 | 0.6830 | 793.6 | 0.0 |
| iALS | Latent factor | 0.0759 | 0.0919 | 0.3464 | 0.0467 | 0.1265 | 0.0375 | 0.0982 | 0.1592 | 0.2756 | 2.9626 | 0.7242 | 941.9 | 13.0 |
| NeuMF | Deep (MLP) | 0.0734 | 0.0857 | 0.3484 | 0.0478 | 0.1249 | 0.0353 | 0.0940 | 0.1485 | 0.3997 | 2.9584 | 0.7095 | 1013.1 | 682.1 |
| Mult-VAE | Deep (VAE) | 0.0757 | 0.0929 | 0.3462 | 0.0484 | 0.1232 | 0.0367 | 0.0970 | 0.1564 | 0.4291 | 2.9810 | 0.7104 | 1003.4 | 826.1 |
| LightGCN | Graph neural network | 0.0771 | 0.0912 | 0.3530 | 0.0492 | 0.1301 | 0.0374 | 0.0984 | 0.1555 | 0.4505 | 3.0528 | 0.7073 | 974.5 | 2440.3 |
| SASRec | Sequential (Transformer) | 0.1506 | 0.1655 | 0.5510 | 0.0956 | 0.2466 | 0.0808 | 0.1808 | 0.2623 | 0.5330 | 3.7169 | 0.6273 | 645.5 | 2563.0 |
| SASRec (CE loss) | Sequential (Transformer) | 0.1784 | 0.1937 | 0.5892 | 0.1099 | 0.2831 | 0.1010 | 0.2108 | 0.2977 | **0.6056** | **3.9569** | 0.6019 | 574.6 | 6178.7 |
| Red Box hybrid | Two-stage hybrid (ours) | **0.1988** | **0.2140** | **0.6444** | **0.1209** | **0.3143** | **0.1147** | **0.2282** | **0.3096** | 0.3401 | 3.5693 | 0.6308 | 697.7 | 28.1 |
| hybrid_mmr0.9 | Two-stage hybrid (ours) | 0.1935 | 0.2090 | 0.6410 | 0.1177 | 0.3081 | 0.1107 | 0.2254 | 0.3090 | 0.3329 | 3.5251 | 0.6841 | 716.0 | 28.1 |
| hybrid_mmr0.8 | Two-stage hybrid (ours) | 0.1876 | 0.2003 | 0.6341 | 0.1144 | 0.3068 | 0.1060 | 0.2209 | 0.3042 | 0.3245 | 3.4806 | 0.7364 | 737.5 | 28.1 |
| hybrid_mmr0.7 | Two-stage hybrid (ours) | 0.1777 | 0.1907 | 0.6203 | 0.1055 | 0.3009 | 0.0990 | 0.2137 | 0.3003 | 0.3152 | 3.4249 | 0.7857 | 764.4 | 28.1 |
| hybrid_mmr0.5 | Two-stage hybrid (ours) | 0.1561 | 0.1607 | 0.5846 | 0.0903 | 0.2871 | 0.0850 | 0.1866 | 0.2580 | 0.2964 | 3.3372 | **0.8581** | 812.7 | 28.1 |
