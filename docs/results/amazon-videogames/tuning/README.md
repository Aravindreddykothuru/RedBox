# Hyperparameter tuning: amazon-videogames

Objective: validation ndcg@10 (test data never used). Sampler: TPE; median pruning for iterative models. Budget per model: at most 30 trials or 10 minutes of wall-clock time on the same 4-core laptop CPU. Trial 0 = paper defaults, so 'best' is never below 'default'.

| Model | Trials (complete / pruned) | Default val | Best val | Gain | Best parameters |
|---|---:|---:|---:|---:|---|
| UserKNN | 14 / 0 | 0.0651 | **0.0700** | +7.4% | k=338, shrink=7.7646 |
| ItemKNN | 30 / 0 | 0.0674 | **0.0718** | +6.6% | k=692, shrink=61.0186 |
| EASE | 30 / 0 | 0.0731 | **0.0758** | +3.7% | l2=51.2579 |
| iALS | 1 / 1 | 0.0633 | **0.0633** | +0.0% | dim=64, alpha=10, reg=1 |

## Which hyperparameters mattered (PED-ANOVA importance, sums to 1)

| Model | Importance |
|---|---|
| UserKNN | shrink 0.73, k 0.27 |
| ItemKNN | k 0.87, shrink 0.13 |
