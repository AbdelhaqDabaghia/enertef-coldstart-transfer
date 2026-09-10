"""
null_model_check.py -- two cheap controls for the RevIN-vs-MinMax comparison.

CONTROL 1 (PART=revin) -- RevIN null model.
  RevIN de-normalises with statistics computed on OBSERVED data:
  pred_kW = pred_net * sigma + mu, with mu/sigma from the lookback window.
  Part of the prediction therefore bypasses the learned weights. If the trivial
  predictor "pred_net = 0" (i.e. pred_kW = mu) is already accurate, the margin in
  which the network operates -- and hence the margin in which forgetting can show
  up -- is narrow BY CONSTRUCTION, and the shrinking CL effects under RevIN would
  be an artefact of the framing rather than evidence that forgetting disappears.

CONTROL 2 (PART=minmax) -- pre-adaptation reference in the production regime.
  We never measured what the production model scores on the source retention
  holdout BEFORE any target fine-tuning, so the MinMax retention numbers cannot be
  turned into a degradation -- only into differences between conditions.

Run one part at a time (window tensors are large):
    PART=revin  python -m scripts.null_model_check
    PART=minmax python -m scripts.null_model_check
Appends to Data/results/null_model_check.csv. No training, no GPU.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from coldstart_transfer.features import SEQ_FEATURES

PART = os.environ.get("PART", "revin")
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))    # driver_source.py convention
OUT = os.environ.get("OUT", "Data/results/null_model_check.csv")
LB = 672
EVKW = SEQ_FEATURES.index("ev_kw")
SRC = "Data/lux_source_features.csv"
TGT = "Data/uk_ev_features_full.csv"


def nrmse(pred_kw, true_kw):
    pred_kw = np.clip(np.asarray(pred_kw, float), 0, None)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    return rmse / (float(true_kw.mean()) + 1e-9)


def tail_windows_evkw(path, days):
    """Only what the null model needs: per-window lookback mean/last of ev_kw and y.

    Avoids materialising the (N, 672, 21) tensor entirely."""
    df = pd.read_csv(path, usecols=["timestamp", "ev_kw"])
    df = df.iloc[-(LB + days * 96):].reset_index(drop=True)
    v = df["ev_kw"].to_numpy(np.float64)
    n = len(v) - LB
    mu = np.empty(n); last = np.empty(n); y = np.empty(n)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    for i in range(n):
        mu[i] = (csum[i + LB] - csum[i]) / LB
        last[i] = v[i + LB - 1]
        y[i] = v[i + LB]
    return mu, last, y


def part_revin():
    rows = []
    for path, days, split in [(TGT, HOLD, "target_holdout"),
                              (SRC, RET_DAYS, "source_retention")]:
        mu, last, y = tail_windows_evkw(path, days)
        rows += [dict(control="revin_null", split=split, predictor="mu (RevIN null)",
                      nrmse=round(nrmse(mu, y), 6), n=len(y),
                      mean_true_kw=round(float(y.mean()), 4)),
                 dict(control="revin_null", split=split, predictor="last observed",
                      nrmse=round(nrmse(last, y), 6), n=len(y),
                      mean_true_kw=round(float(y.mean()), 4))]
    return rows


def part_minmax():
    import joblib
    from coldstart_transfer.model import build_ev_model, load_production_weights
    from coldstart_transfer.windowing import make_windows, split_holdout
    from coldstart_transfer.trainer import nrmse as trainer_nrmse

    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")
    rows = []

    # source retention holdout -- same tail driver_source.py windows
    sdf = pd.read_csv(SRC).iloc[-((POOL_DAYS + RET_DAYS) * 96 + LB):].reset_index(drop=True)
    a, b, c, d = make_windows(sdf, scalers)
    _, ho = split_holdout(a, b, c, d, holdout_days=RET_DAYS)
    v, _, mean_kw = trainer_nrmse(m, ho, scalers)
    rows.append(dict(control="minmax_pre_adaptation", split="source_retention",
                     predictor="production model, no fine-tune",
                     nrmse=round(v, 6), n=len(ho[2]), mean_true_kw=round(mean_kw, 4)))
    del a, b, c, d, ho

    # target holdout -- the last HOLD days only (same rows split_holdout would pick)
    tdf = pd.read_csv(TGT).iloc[-(LB + HOLD * 96):].reset_index(drop=True)
    a, b, c, d = make_windows(tdf, scalers)
    v, _, mean_kw = trainer_nrmse(m, (a, b, c, d), scalers)
    rows.append(dict(control="minmax_pre_adaptation", split="target_holdout",
                     predictor="production model, no fine-tune (zero-shot)",
                     nrmse=round(v, 6), n=len(c), mean_true_kw=round(mean_kw, 4)))
    return rows


def main():
    rows = part_revin() if PART == "revin" else part_minmax()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(df.to_string(index=False))
    print(f"[null] appended {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
