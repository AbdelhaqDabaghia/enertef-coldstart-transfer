"""
e19_train_serve_gap.py -- how is the deployed model ACTUALLY performing?

E7 compared two feature conventions. Neither is what the service computes.

    train   roll_w[t] = mean(y[t-w+1 .. t])        <- what the model was fitted on
    causal  roll_w[t] = mean(y[t-w  .. t-1])       <- what a correct pipeline does
    serve   roll_w[t] = sum(y[t-w .. t-1]) / w     <- what production feeds

`serve` is the deployed arithmetic: ev_features_builder.compute_lag_features
takes the window inclusive of the current index, and realtime_runner calls it
through build_ev_next_input BEFORE that step is predicted, on an array whose
forecast slots are still zero. The service therefore divides by w while summing
w-1 real values -- (w-1)/w of the causal value, so 25 % low on the 1 h window.

The model is unchanged across all three arms. Any difference is the features.
The `serve` row is the number the site has actually been getting.

Writes Data/results/corrected_causal_pipeline/e19_train_serve_gap.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

from coldstart_transfer.features import engineer_features, WEATHER_VARS
from coldstart_transfer.windowing import (make_windows, split_holdout,
                                          inverse_target, LOOKBACK)
from coldstart_transfer.model import build_ev_model, load_production_weights

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
LUX = "Data/lux_source_features.csv"
OUT = os.environ.get(
    "OUT", "Data/results/corrected_causal_pipeline/e19_train_serve_gap.csv")
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
MODES = ["train", "causal", "serve"]


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (t.mean() + 1e-9))


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    raw = pd.read_csv(LUX)[["timestamp", "ev_kw"] + WEATHER_VARS]
    m = build_ev_model()
    load_production_weights(m, PROD)

    rows = []
    for mode in MODES:
        feat = engineer_features(raw, mode=mode)
        tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
        a_, b_, c_, d_ = make_windows(feat.iloc[-tail:].reset_index(drop=True),
                                      scalers)
        pool, ho = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

        Xs, Xn, y, _ = ho
        t = inverse_target(y, scalers)
        p = np.clip(inverse_target(
            np.clip(m.predict([Xs, Xn], verbose=0).reshape(-1), 0, None),
            scalers), 0, None)

        Xsp, Xnp, yp, _ = pool
        tp = inverse_target(yp, scalers)
        pp = np.clip(inverse_target(
            np.clip(m.predict([Xsp, Xnp], verbose=0).reshape(-1), 0, None),
            scalers), 0, None)
        a = float((pp @ tp) / (pp @ pp + 1e-12))

        pers = np.concatenate([[t[0]], t[:-1]])
        rows.append(dict(
            mode=mode,
            nrmse=round(nrmse(p, t), 4),
            nrmse_gauged=round(nrmse(a * p, t), 4),
            gauge_a=round(a, 4),
            corr=round(float(np.corrcoef(p, t)[0, 1]), 4),
            bias_kw=round(float((p - t).mean()), 2),
            pred_mean_kw=round(float(p.mean()), 2),
            true_mean_kw=round(float(t.mean()), 2),
            persistence=round(nrmse(pers, t), 4)))
        print("[e19] %-7s nRMSE=%.4f gauged=%.4f a=%.3f corr=%.3f "
              "bias=%+.1f kW  (pred %.1f vs true %.1f)"
              % (mode, rows[-1]["nrmse"], rows[-1]["nrmse_gauged"], a,
                 rows[-1]["corr"], rows[-1]["bias_kw"],
                 rows[-1]["pred_mean_kw"], rows[-1]["true_mean_kw"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n" + df.to_string(index=False))

    tr = df[df["mode"] == "train"].iloc[0]
    sv = df[df["mode"] == "serve"].iloc[0]
    print("\n  what the site actually gets vs what training suggested:")
    print("    nRMSE  %.4f -> %.4f  (%.2fx worse)"
          % (tr["nrmse"], sv["nrmse"], sv["nrmse"] / tr["nrmse"]))
    print("    corr   %.3f -> %.3f" % (tr["corr"], sv["corr"]))
    print("    bias   %+.1f -> %+.1f kW" % (tr["bias_kw"], sv["bias_kw"]))
    print("    persistence on the same holdout: %.4f" % tr["persistence"])
    print("\n[e19] wrote %s" % OUT)


if __name__ == "__main__":
    main()
