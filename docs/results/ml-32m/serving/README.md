# Served hybrid vs its models: ml-32m

The hybrid the app serves and each of its candidate generators, evaluated on the same 15,000 sampled test users as the export (same seed). Full-catalogue ranking; the test set is used once. Reproduce: `recsys serving-eval -c ml-32m`.

| model | family | NDCG@10 | Recall@10 | HR@10 | P@10 | MRR@10 | MAP@10 | NDCG@20 | Recall@20 | Coverage@10 | Novelty@10 | ILD@10 | ARP@10 | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red Box hybrid | Two-stage hybrid (ours) | **0.1293** | **0.1359** | **0.4841** | **0.0776** | **0.2262** | **0.0703** | **0.1514** | **0.2086** | 0.4251 | 4.0078 | 0.7074 | 4410.7 | n/a |
| EASE | Linear autoencoder | 0.0913 | 0.1012 | 0.3849 | 0.0566 | 0.1602 | 0.0467 | 0.1114 | 0.1640 | 0.2068 | 2.8803 | 0.7529 | 7241.7 | 43.4 |
| SASRec (CE loss) | Sequential (Transformer) | 0.1115 | 0.1206 | 0.4275 | 0.0659 | 0.1942 | 0.0602 | 0.1325 | 0.1876 | **0.5274** | **4.3926** | 0.6941 | 3635.7 | 5327.6 |
| Markov chain | Sequential (Markov) | 0.0735 | 0.0809 | 0.2934 | 0.0401 | 0.1317 | 0.0401 | 0.0864 | 0.1218 | 0.4801 | 3.9614 | 0.7512 | 4476.5 | 0.3 |
| iALS | Latent factor | 0.0810 | 0.0916 | 0.3542 | 0.0500 | 0.1425 | 0.0407 | 0.1005 | 0.1498 | 0.1526 | 2.9845 | 0.7423 | 6569.7 | 182.8 |
| MostPop | Baseline | 0.0377 | 0.0398 | 0.1814 | 0.0249 | 0.0712 | 0.0179 | 0.0485 | 0.0726 | 0.0126 | 1.6632 | **0.7840** | 13592.0 | 0.0 |

## Significance (Wilcoxon signed-rank on per-user NDCG@10, Holm-corrected)

| comparison | relative difference | users better / worse / tied | p-value (Holm) |
|---|---:|---:|---:|
| Red Box hybrid vs MostPop | +243.1% | 6559 / 1532 / 6909 | 0.0e+00 |
| Red Box hybrid vs Markov chain | +76.0% | 5821 / 1912 / 7267 | 0.0e+00 |
| Red Box hybrid vs iALS | +59.6% | 5477 / 2830 / 6693 | 1.3e-241 |
| Red Box hybrid vs EASE | +41.6% | 5203 / 3122 / 6675 | 2.2e-163 |
| Red Box hybrid vs SASRec (CE loss) | +15.9% | 4507 / 2926 / 7567 | 1.3e-87 |
