"""
e13_gauge_decomposition.py -- is replay's retention advantage entirely gauge?

THE PUZZLE. Under the deployed MinMax representation, two routes reach the same
retention number:
    B5_replay  0.4293   -- re-injects 60 days of SOURCE data
    freeze3    0.4196   -- sees NO source data, only re-trains the head
    (difference -0.0097, p = 0.557: indistinguishable)
while plain fine-tuning sits at 0.5572 and the un-adapted model at 0.5479.

One route has access to the source distribution and the other has none, yet they
land together. The parsimonious reading is that neither retains knowledge: both
restore an amplitude. E4 supports it -- the production model under-predicts its
own source domain by a factor 1.37, and a single scalar fitted OUT OF SAMPLE
removes 28 % of its error (0.5479 -> 0.3961, 92.8 % of the in-sample gain).

THE TEST. For each adapted model, fit the one-parameter gauge a on the SOURCE
POOL (never used for evaluation) and apply it unchanged to the retention
holdout. Then re-compare the conditions.

    if B3_warm-gauged reaches B5_replay-gauged
        -> replay's retention advantage IS the gauge. "Replay retains better"
           comes out of the paper and is replaced by "replay re-learns an
           amplitude that one scalar supplies for free".
    if a gap survives
        -> there is a structural component; quantify it. That residual is the
           only thing worth calling retention.

Reports, per condition: raw retention, gauged retention, the fitted a, and the
decomposition of the gap to the pre-adaptation reference into an amplitude part
and a structural part.

Writes Data/results/e13_gauge_decomposition.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.trainer import run_condition
from coldstart_transfer.cl_methods import run_cl_method
from coldstart_transfer.logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = os.environ.get("OUT", "Data/results/e13_gauge_decomposition.csv")
N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def preds_kw(model, part, scalers):
    Xs, Xn, y, _ = part
    p = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
    return np.clip(inverse_target(p, scalers), 0, None), inverse_target(y, scalers)


def nrmse(p, t):
    return float(np.sqrt(np.mean((np.clip(p, 0, None) - t) ** 2))) / (float(t.mean()) + 1e-9)


def gauge_from_pool(model, pool, ho, scalers):
    """Fit the single-parameter gauge on the pool, apply it to the holdout."""
    p_pool, t_pool = preds_kw(model, pool, scalers)
    p_ho, t_ho = preds_kw(model, ho, scalers)
    a = float((p_pool @ t_pool) / (p_pool @ p_pool + 1e-12))
    return nrmse(p_ho, t_ho), nrmse(a * p_ho, t_ho), a


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)

    # the un-adapted reference, gauged the same way
    keras.backend.clear_session()
    m0 = build_ev_model()
    load_production_weights(m0, PROD)
    r0, g0, a0 = gauge_from_pool(m0, source_pool, source_holdout, scalers)
    print(f"[e13] reference (no adaptation): raw={r0:.4f} gauged={g0:.4f} a={a0:.3f}",
          flush=True)
    rows = [dict(condition="reference_no_adaptation", seed=-1, raw=round(r0, 6),
                 gauged=round(g0, 6), a=round(a0, 4))]

    for seed in SEEDS:
        for name in ["B3_warm", "MAS", "B5_replay"]:
            keras.backend.clear_session()
            if name == "MAS":
                r = run_cl_method("MAS", target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                                  batch_size=BATCH, lr=LR, source_train=source_pool,
                                  source_holdout=source_holdout)
            else:
                base = "B5" if name == "B5_replay" else "B3"
                r = run_condition(base, target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, lambda_ewc=0.0,
                                  beta_replay=1.0 if base == "B5" else 0.0,
                                  seed=seed, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  source_train=source_pool,
                                  source_holdout=source_holdout)
            raw, gauged, a = gauge_from_pool(r["model"], source_pool,
                                             source_holdout, scalers)
            rows.append(dict(condition=name, seed=seed, raw=round(raw, 6),
                             gauged=round(gauged, 6), a=round(a, 4),
                             target=float(r["target_nrmse"])))
            print(f"[e13 {name} s={seed}] raw={raw:.4f} gauged={gauged:.4f} a={a:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== retention, raw vs gauged (mean over seeds) ===")
    g = df[df.seed >= 0].groupby("condition")[["raw", "gauged", "a"]].mean()
    g["gap_closed_%"] = 100 * (g.raw - g.gauged) / g.raw
    print(g.round(4).to_string())
    if {"B3_warm", "B5_replay"} <= set(g.index):
        print(f"\n  B3 - replay, raw    : {g.loc['B3_warm','raw'] - g.loc['B5_replay','raw']:+.4f}")
        print(f"  B3 - replay, gauged : {g.loc['B3_warm','gauged'] - g.loc['B5_replay','gauged']:+.4f}")
        print("  If the gauged gap collapses, replay's advantage was amplitude.")
    print(f"\n[e13] wrote {OUT}")


if __name__ == "__main__":
    main()
