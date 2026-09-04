"""
baseline_persistence.py -- naive / seasonal-naive reference forecasters.

Answers the standard reviewer question "did you beat persistence?" by scoring
model-free baselines on the *exact same* 14-day target holdout the trained models
are evaluated on (windowing.split_holdout, holdout_days=14 -> last 14*96=1344
predicted steps), with the *same* metric as trainer.nrmse:

    nRMSE = RMSE / mean(true_kw),  predictions clipped >= 0,  computed in kW.

No model, no GPU, no scaler round-trip: MinMax transform+inverse is the identity,
so the raw target column IS the un-scaled truth trainer.nrmse compares against.
This keeps the baseline honest and independent of the training path.

Baselines (forecast of the target at step t):
    naive-1        y_hat(t) = y(t-1)     (last observation, 15 min ago)
    seasonal-day   y_hat(t) = y(t-96)    (same time yesterday)
    seasonal-week  y_hat(t) = y(t-672)   (same time last week)

Run from the repo root:
    python -m coldstart_transfer.baseline_persistence
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HOLDOUT_DAYS = 14
STEPS_PER_DAY = 96                      # 15-min resolution
HOLDOUT_STEPS = HOLDOUT_DAYS * STEPS_PER_DAY   # 1344, matches split_holdout
OUT = "Data/results/persistence_baseline.csv"

# (name, csv, target column) -- same target files the drivers use.
TASKS = [
    ("EV (LU->UK)", "Data/uk_ev_features_full.csv", "ev_kw"),
    ("PV (LU->Konstanz)", "Data/pv_target/konstanz_pv_features.csv", "pv_kw"),
]

# lag in steps for each persistence variant
LAGS = {"naive-1": 1, "seasonal-day": 96, "seasonal-week": 672}


def score(true_kw: np.ndarray, pred_kw: np.ndarray):
    """Replicate trainer.nrmse exactly: clip preds >=0, RMSE in kW, nRMSE=RMSE/mean."""
    pred_kw = np.clip(pred_kw, 0, None)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    mean = float(np.mean(true_kw))
    nrmse = rmse / mean if mean > 1e-9 else float("nan")
    return rmse, nrmse


def run_task(name: str, csv: str, col: str) -> list[dict]:
    df = pd.read_csv(csv)
    assert col in df.columns, f"{csv}: missing target column {col!r}"
    y = df[col].to_numpy(dtype=np.float64)
    n = len(y)
    # predicted steps of the final 14-day holdout == the last HOLDOUT_STEPS rows
    # of the frame (window i predicts df row i+LOOKBACK, so the last 1344 windows
    # predict the last 1344 rows -- independent of LOOKBACK).
    ho_idx = np.arange(n - HOLDOUT_STEPS, n)
    true_kw = y[ho_idx]
    rows = []
    print(f"\n=== {name} | holdout rows {ho_idx[0]}..{ho_idx[-1]} "
          f"({len(ho_idx)} steps) | mean(true)={true_kw.mean():.3f} kW ===")
    for variant, lag in LAGS.items():
        pred_kw = y[ho_idx - lag]
        rmse, nrmse = score(true_kw, pred_kw)
        rows.append({"task": name, "baseline": variant, "lag_steps": lag,
                     "holdout_steps": len(ho_idx),
                     "rmse_kw": round(rmse, 4), "nrmse": round(nrmse, 6),
                     "mean_true_kw": round(float(true_kw.mean()), 4)})
        print(f"  {variant:<14} lag={lag:<4}  RMSE={rmse:8.4f} kW   nRMSE={nrmse:.4f}")
    return rows


def main():
    all_rows = []
    for name, csv, col in TASKS:
        all_rows += run_task(name, csv, col)
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT, index=False)
    print(f"\n[persistence] wrote {len(out)} rows -> {OUT}")
    # convenience: best seasonal-naive per task (the honest 'strong naive' to beat)
    for name in out.task.unique():
        sub = out[(out.task == name) & (out.baseline != "naive-1")]
        best = sub.loc[sub.nrmse.idxmin()]
        print(f"[persistence] {name}: strongest naive = {best.baseline} "
              f"(RMSE={best.rmse_kw} kW, nRMSE={best.nrmse})")


if __name__ == "__main__":
    main()
