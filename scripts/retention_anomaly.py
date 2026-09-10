"""
retention_anomaly.py -- why does a FROZEN-trunk model fine-tuned on UK data score
BETTER on the Luxembourg source holdout than the production model it started from?

Observed (MinMax regime, source retention holdout, 10 seeds):
    production model, no fine-tune ....... 0.5479
    B3 full fine-tune on UK .............. 0.5572   (+1.7%)
    freeze3 (trunk frozen), UK fine-tune . 0.4196   (-23.4%)   <-- anomaly

Hypothesis H1 (calibration): the production model is mis-calibrated on this
holdout -- a systematic scale/offset error. freeze3 keeps the trunk (hence the
learned features) but re-trains the head, which can absorb an affine correction.
If so, simply refitting the best affine map a*pred + b on the production model's
predictions should recover most of the gap, and the "retention" metric is largely
measuring output calibration rather than knowledge preservation.

Hypothesis H2 (evaluation bug): the retention probe does not evaluate what we
think. Checked here by re-deriving the holdout independently and comparing.

Prints a decomposition; writes nothing.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target

POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
LB = 672
SRC = "Data/lux_source_features.csv"


def nrmse(pred, true):
    pred = np.clip(pred, 0, None)
    return float(np.sqrt(np.mean((pred - true) ** 2))) / (float(true.mean()) + 1e-9)


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")

    sdf = pd.read_csv(SRC).iloc[-((POOL_DAYS + RET_DAYS) * 96 + LB):].reset_index(drop=True)
    a, b, c, d = make_windows(sdf, scalers)
    _, ho = split_holdout(a, b, c, d, holdout_days=RET_DAYS)
    Xs, Xn, y_s, _ = ho

    pred_s = m.predict([Xs, Xn], verbose=0).reshape(-1)
    pred_kw = np.clip(inverse_target(pred_s, scalers), 0, None)
    true_kw = inverse_target(y_s, scalers)

    print(f"holdout windows       : {len(true_kw)}")
    print(f"true  kW  mean/std    : {true_kw.mean():8.3f} / {true_kw.std():8.3f}")
    print(f"pred  kW  mean/std    : {pred_kw.mean():8.3f} / {pred_kw.std():8.3f}")
    print(f"bias (pred-true) mean : {(pred_kw - true_kw).mean():+8.3f}")
    print(f"corr(pred,true)       : {np.corrcoef(pred_kw, true_kw)[0,1]:8.4f}")
    print()

    base = nrmse(pred_kw, true_kw)
    print(f"[as-is]                nRMSE = {base:.4f}")

    # --- H1: best affine recalibration a*pred + b (fit ON the holdout: an upper
    #     bound on what any output-layer re-fit could achieve here) ---
    A = np.c_[pred_kw, np.ones_like(pred_kw)]
    coef, *_ = np.linalg.lstsq(A, true_kw, rcond=None)
    recal = A @ coef
    print(f"[affine recalibrated]  nRMSE = {nrmse(recal, true_kw):.4f}   "
          f"(a={coef[0]:.3f}, b={coef[1]:+.3f})")

    # scale-only and offset-only, to see which part matters
    s = float((pred_kw @ true_kw) / (pred_kw @ pred_kw))
    print(f"[scale-only  a*pred]   nRMSE = {nrmse(s * pred_kw, true_kw):.4f}   (a={s:.3f})")
    off = float((true_kw - pred_kw).mean())
    print(f"[offset-only pred+b]   nRMSE = {nrmse(pred_kw + off, true_kw):.4f}   (b={off:+.3f})")
    print()
    print(f"freeze3 measured       nRMSE = 0.4196   (target to explain)")
    print(f"B3 full fine-tune      nRMSE = 0.5572")

    # --- trivial references on the same holdout ---
    ev = sdf["ev_kw"].to_numpy(float)
    last = ev[LB - 1:-1][-len(true_kw):]
    print()
    print(f"[persistence last obs] nRMSE = {nrmse(last, true_kw):.4f}")
    print(f"[constant = mean]      nRMSE = {nrmse(np.full_like(true_kw, true_kw.mean()), true_kw):.4f}")


if __name__ == "__main__":
    main()
