# Pre-adaptation reference (the starting point of every retention table)

_Scores of the transferred production model BEFORE any target fine-tuning,
on exactly the holdouts the drivers evaluate on. A condition scoring BELOW
the source-retention reference has not preserved the source: it has improved
on it. See `scripts/retention_anomaly.py` -- on EV most of that improvement is
amplitude re-calibration (corr stays 0.958 while the model under-predicts by
15 kW), not knowledge retention._

| task | split | nRMSE | bias (kW) | corr | mean true (kW) | n |
|---|---|---:|---:|---:|---:|---:|
| EV (LU->UK) | source_retention | **0.5479** | -15.17 | 0.958 | 56.92 | 672 |
| EV (LU->UK) | target_zero_shot | **0.7964** | -9.71 | 0.602 | 23.03 | 1344 |
| PV (LU->Konstanz) | source_retention | **0.6126** | -0.88 | 0.944 | 16.80 | 672 |
| PV (LU->Konstanz) | target_zero_shot | **66.4810** | +27.30 | 0.938 | 0.67 | 1344 |

Reference for the RevIN regime (source model retrained in-representation,
see scripts/driver_freeze_revin.py): source retention **0.2034** before
adaptation, **0.2355** after full fine-tuning (+15.8%).
