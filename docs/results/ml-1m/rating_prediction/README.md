# Rating prediction (experiment E7): ml-1m

Every rating counts (no 'liked' threshold). Split `all_val0.1_test0.1_min3_seed42` (fingerprint `dc679ea69ddb2d22`): train/val/test ratings = 799,663 / 100,253 / 100,237, same per-user chronological protocol as the ranking experiments. Predictions are clipped to 1-5 stars. Lower is better; best test RMSE in **bold**.

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE | Fit (s) |
|---|---:|---:|---:|---:|---:|
| Global mean | 1.1291 | 0.9302 | 1.1620 | 0.9588 | 0.0 |
| User + item biases | 0.9165 | 0.7222 | 0.9272 | 0.7307 | 0.5 |
| Biased MF (SVD) | 0.8574 | 0.6706 | **0.8634** | 0.6751 | 95.3 |
