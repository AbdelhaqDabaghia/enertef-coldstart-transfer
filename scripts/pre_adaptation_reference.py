"""
pre_adaptation_reference.py -- the missing row of every retention table.

Each retention table compares conditions against each other but never against the
STARTING POINT: what the transferred model scores before any target fine-tuning.
Without it, a number like "B5 retention = 0.4293" cannot be read as preservation
(it is in fact BETTER than the 0.5479 the model started from -- see
scripts/retention_anomaly.py, which shows most of that gap is amplitude
re-calibration, not knowledge retention).

This computes, for each task/regime:
  - source-retention nRMSE before adaptation  (the retention reference)
  - target nRMSE before adaptation            (zero-shot transfer)
  - bias and correlation on the source holdout, so calibration error can be told
    apart from loss of signal shape.

Holdout conventions are read from the drivers they must match:
  EV  : tail (POOL_DAYS+RET_DAYS)*96 + 672, split_holdout(RET_DAYS)   <- identical in
        driver_source / driver_ewc_bench / driver_cl_methods / driver_freeze_sweep
  PV  : same, with PV_LOOKBACK                                        <- driver_pv_source
Target holdout is the fixed final 14 days in both cases.

Writes Data/results/pre_adaptation_reference.csv and a markdown block ready to
paste above each retention table.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
HOLD = int(os.environ.get("HOLD", "14"))
OUT = os.environ.get("OUT", "Data/results/pre_adaptation_reference.csv")
MD = os.environ.get("MD", "Data/results/stats/pre_adaptation_reference.md")


def scores(model, holdout, scalers, inverse):
    Xs, Xn, y, _ = holdout
    pred = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
    pred_kw = np.clip(inverse(pred, scalers), 0, None)
    true_kw = inverse(y, scalers)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    mean = float(true_kw.mean())
    return dict(nrmse=round(rmse / (mean + 1e-9), 6),
                rmse_kw=round(rmse, 4), mean_true_kw=round(mean, 4),
                bias_kw=round(float((pred_kw - true_kw).mean()), 4),
                corr=round(float(np.corrcoef(pred_kw, true_kw)[0, 1]), 4),
                n=len(true_kw))


def ev_rows():
    from coldstart_transfer.model import build_ev_model, load_production_weights
    from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")
    rows = []

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a, b, c, d = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    _, sho = split_holdout(a, b, c, d, holdout_days=RET_DAYS)
    rows.append(dict(task="EV (LU->UK)", regime="MinMax (production scalers)",
                     split="source_retention", **scores(m, sho, scalers, inverse_target)))
    del a, b, c, d, sho

    tdf = pd.read_csv("Data/uk_ev_features_full.csv").iloc[-(LOOKBACK + HOLD * 96):]
    a, b, c, d = make_windows(tdf.reset_index(drop=True), scalers)
    rows.append(dict(task="EV (LU->UK)", regime="MinMax (production scalers)",
                     split="target_zero_shot", **scores(m, (a, b, c, d), scalers, inverse_target)))
    return rows


def pv_rows():
    from coldstart_transfer.pv import (build_pv_model, load_pv_production_weights,
                                       make_pv_windows, PV_LOOKBACK)
    from coldstart_transfer.windowing import split_holdout, inverse_target
    scalers = joblib.load("Data/models/pv_scalers.joblib")
    m = build_pv_model()
    load_pv_production_weights(m, "Data/models/pv_lstm_openmeteo.keras")
    rows = []

    sdf = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + PV_LOOKBACK
    a, b, c, d = make_pv_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    _, sho = split_holdout(a, b, c, d, holdout_days=RET_DAYS)
    rows.append(dict(task="PV (LU->Konstanz)", regime="MinMax (production scalers)",
                     split="source_retention", **scores(m, sho, scalers, inverse_target)))
    del a, b, c, d, sho

    tdf = pd.read_csv("Data/pv_target/konstanz_pv_features.csv")
    a, b, c, d = make_pv_windows(tdf, scalers)
    _, tho = split_holdout(a, b, c, d, holdout_days=HOLD)
    rows.append(dict(task="PV (LU->Konstanz)", regime="MinMax (production scalers)",
                     split="target_zero_shot", **scores(m, tho, scalers, inverse_target)))
    return rows


def main():
    rows = ev_rows() + pv_rows()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))

    # --- markdown block to paste above each retention table ---
    os.makedirs(os.path.dirname(MD), exist_ok=True)
    L = ["# Pre-adaptation reference (the starting point of every retention table)",
         "",
         "_Scores of the transferred production model BEFORE any target fine-tuning,",
         "on exactly the holdouts the drivers evaluate on. A condition scoring BELOW",
         "the source-retention reference has not preserved the source: it has improved",
         "on it. See `scripts/retention_anomaly.py` -- on EV most of that improvement is",
         "amplitude re-calibration (corr stays 0.958 while the model under-predicts by",
         "15 kW), not knowledge retention._",
         "",
         "| task | split | nRMSE | bias (kW) | corr | mean true (kW) | n |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['split']} | **{r['nrmse']:.4f}** | "
                 f"{r['bias_kw']:+.2f} | {r['corr']:.3f} | {r['mean_true_kw']:.2f} | {r['n']} |")
    L += ["",
          "Reference for the RevIN regime (source model retrained in-representation,",
          "see scripts/driver_freeze_revin.py): source retention **0.2034** before",
          "adaptation, **0.2355** after full fine-tuning (+15.8%).",
          ""]
    open(MD, "w", encoding="utf-8").write("\n".join(L))
    print(f"\n[pre-adapt] wrote {OUT} and {MD}")


if __name__ == "__main__":
    main()
