# Paper Graph Values

Values are mean +/- std across saved runs.

| Model | Config | Accuracy | Precision | Recall | F1 | Client variance |
|---|---:|---:|---:|---:|---:|---:|
| Fusion | IID | 84.86 +/- 1.48 | 85.24 +/- 1.39 | 84.86 +/- 1.48 | 84.77 +/- 1.59 | 161.02 +/- 94.50 |
| Fusion | Non-IID + DP | 83.84 +/- 8.59 | 78.65 +/- 8.88 | 83.84 +/- 8.59 | 79.45 +/- 8.59 | 667.95 +/- 490.39 |

## Fusion Per Seed

| Seed | IID accuracy | Non-IID + DP accuracy |
|---:|---:|---:|
| 42 | 83.78 | 81.08 |
| 123 | 86.49 | 70.00 |
| 456 | 83.78 | 90.91 |
| 789 | 86.49 | 89.19 |
| 2024 | 83.78 | 88.00 |
