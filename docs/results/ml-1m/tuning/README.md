# Hyperparameter tuning: ml-1m

Objective: validation ndcg@10 (test data never used). Sampler: TPE; median pruning for iterative models. Budget per model: at most 40 trials or 25 minutes of wall-clock time on the same 4-core laptop CPU. Trial 0 = paper defaults, so 'best' is never below 'default'.

| Model | Trials (complete / pruned) | Default val | Best val | Gain | Best parameters |
|---|---:|---:|---:|---:|---|
| UserKNN | 40 / 0 | 0.0792 | **0.0810** | +2.3% | k=46, shrink=30.0814 |
| ItemKNN | 40 / 0 | 0.0765 | **0.0787** | +3.0% | k=400, shrink=4.3433 |
| EASE | 40 / 0 | 0.0867 | **0.0868** | +0.1% | l2=509.9258 |
| BPR-MF | 6 / 7 | 0.0709 | **0.0801** | +13.0% | dim=128, lr=0.00164, reg=0.005242 |
| iALS | 2 / 1 | 0.073 | **0.0759** | +3.9% | dim=64, alpha=4.7186, reg=16.9617 |
| NeuMF | 3 / 5 | 0.0716 | **0.0734** | +2.6% | gmf_dim=32, mlp_dims=[128, 64, 32], negatives=8, lr=0.0005846 |
| Mult-VAE | 5 / 5 | 0.0757 | **0.0757** | +0.0% | dropout=0.5, anneal_cap=0.2, anneal_epochs=100, lr=0.001 |

## Which hyperparameters mattered (PED-ANOVA importance, sums to 1)

| Model | Importance |
|---|---|
| UserKNN | shrink 0.59, k 0.41 |
| ItemKNN | shrink 0.82, k 0.18 |
