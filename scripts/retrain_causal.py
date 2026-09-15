"""
retrain_causal.py -- refit the EV model on the causal convention.

WHY. The deployed model was fitted on features whose rolling means contain
y_t, and is served features that contain neither y_t nor a correct average of
the known values (e19). Measured on the retention holdout, the model the site
actually runs predicts a mean of 14.5 kW where 56.9 kW arrives.

There is no serving-side-only repair (tests/test_serving_parity.py pins this).
One convention has to be chosen, implemented on both sides, and the model
refitted. This script does the refitting half, on `causal`:

    roll_w[t] = mean(y[t-w .. t-1])

every feature available at t depending only on y_(t-1) and earlier.

SCALERS. The frozen production scalers are re-fitted here, deliberately and
unlike everywhere else in this repository. Elsewhere we freeze them to keep
weight compatibility with the transferred artefact; here we are replacing that
artefact, and the causal rolling columns have a different support, so keeping
the old ranges would push part of the input outside [0, 1]. The new scalers are
written alongside the model and the two must be deployed together.

ARMS. --arms selects the initialisations to fit: from scratch, warm-started
from the deployed weights, or both. Every arm is reported against naive
persistence on the same holdout, because after e19 an absolute floor matters
more than a relative improvement.

A 120-day, 12-epoch signal run gave, against persistence at 0.9972:
    scratch  0.9532   pred mean 42.4 kW
    warm     0.9035   pred mean 44.8 kW      <- warm wins
    (deployed today, served features: 1.3890, pred mean 14.5 kW; true 56.9 kW)
so the deployed weights carry transferable structure despite having been fitted
under a convention that no longer applies, and a production run may spend its
budget on `warm` alone.

Writes Data/models/ev_cnn_lstm_causal_<tag>.keras, the matching scalers, and
Data/results/corrected_causal_pipeline/retrain_causal.csv.
"""
from __future__ import annotations
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from coldstart_transfer.features import engineer_features, SEQ_FEATURES, \
    NEXT_EXO, WEATHER_VARS
from coldstart_transfer.windowing import make_windows, split_holdout, \
    inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, compile_ev_model, \
    load_production_weights

LUX = "Data/lux_source_features.csv"
PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUTDIR = "Data/models"
# Overridable, because a follow-up single-seed run would otherwise overwrite a
# committed multi-seed campaign with one row.
RESULTS = os.environ.get(
    "RESULTS", "Data/results/corrected_causal_pipeline/retrain_causal.csv")

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def fit_scalers(df: pd.DataFrame, train_rows: int) -> dict:
    """Fit on the TRAINING rows only. Fitting on everything would leak the
    holdout's range into the transform."""
    tr = df.iloc[:train_rows]
    s_seq = MinMaxScaler().fit(tr[SEQ_FEATURES].to_numpy(np.float32))
    s_next = MinMaxScaler().fit(tr[NEXT_EXO].to_numpy(np.float32))
    s_y = MinMaxScaler().fit(tr[["ev_kw"]].to_numpy(np.float32))
    return {"scaler_seq": s_seq, "scaler_next": s_next, "scaler_y": s_y}


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (t.mean() + 1e-9))


def evaluate(model, part, scalers):
    Xs, Xn, y, _ = part
    t = inverse_target(y, scalers)
    p = np.clip(inverse_target(
        np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None),
        scalers), 0, None)
    pers = np.concatenate([[t[0]], t[:-1]])
    return dict(nrmse=nrmse(p, t), mae=float(np.mean(np.abs(p - t))),
                bias_kw=float((p - t).mean()), pred_mean=float(p.mean()),
                true_mean=float(t.mean()), persistence=nrmse(pers, t),
                corr=float(np.corrcoef(p, t)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="causal",
                    choices=["causal", "train", "serve"],
                    help="feature convention to fit on")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hold-days", type=int, default=7)
    ap.add_argument("--train-days", type=int, default=0,
                    help="cap the training window to the most recent N days "
                         "(0 = use everything). The full series is ~46k windows "
                         "of 672 steps through an LSTM, which is slow; capping "
                         "buys a signal quickly. It changes how much history "
                         "the model sees, so report the cap with the result.")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="scratch,warm",
                    help="which initialisations to fit. The signal run showed "
                         "warm-starting from the deployed weights beats "
                         "scratch (0.9035 vs 0.9532), so a production run can "
                         "spend its budget on 'warm' alone -- but keep both "
                         "whenever the claim itself is being tested.")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ("scratch", "warm")]
    if bad:
        raise SystemExit("unknown arm(s): %s" % bad)

    raw = pd.read_csv(LUX)[["timestamp", "ev_kw"] + WEATHER_VARS]
    feat = engineer_features(raw, mode=a.mode)
    if a.train_days:
        # keep the LOOKBACK warm-up ahead of the capped window, or the first
        # windows would carry truncated lags
        keep = (a.train_days + a.hold_days) * 96 + LOOKBACK
        feat = feat.iloc[-keep:].reset_index(drop=True)
    print("[retrain] %d rows, convention '%s'%s"
          % (len(feat), a.mode,
             "" if not a.train_days else
             " (capped to %d training days)" % a.train_days), flush=True)

    # scalers on the training portion only
    n_hold = a.hold_days * 96
    scalers = fit_scalers(feat, len(feat) - n_hold - LOOKBACK)
    Xs, Xn, y, ts = make_windows(feat, scalers)
    train, hold = split_holdout(Xs, Xn, y, ts, holdout_days=a.hold_days)
    print("[retrain] train %d windows | holdout %d (%d days)"
          % (len(train[2]), len(hold[2]), a.hold_days), flush=True)

    rows = []
    for seed in seeds:
        for arm in arms:
            keras.backend.clear_session()
            if arm == "scratch":
                m = build_ev_model(seed=seed)
            else:
                # warm start is only meaningful if the topology matches; the
                # weights were fitted on a different convention, so this tests
                # whether the old features are a useful starting point at all
                m = build_ev_model(seed=seed)
                load_production_weights(m, PROD)
            compile_ev_model(m, lr=a.lr)
            m.fit([train[0], train[1]], train[2], epochs=a.epochs,
                  batch_size=a.batch, verbose=0, shuffle=True)

            r = evaluate(m, hold, scalers)
            r.update(mode=a.mode, arm=arm, seed=seed, epochs=a.epochs)
            rows.append(r)
            print("[retrain %-7s s=%d] nRMSE=%.4f (persistence %.4f) "
                  "bias=%+.1f kW  pred %.1f vs true %.1f  corr=%.3f  %s"
                  % (arm, seed, r["nrmse"], r["persistence"], r["bias_kw"],
                     r["pred_mean"], r["true_mean"], r["corr"],
                     "BEATS persistence" if r["nrmse"] < r["persistence"]
                     else "WORSE than persistence"), flush=True)

            # one artefact per arm, at the first seed. The tag carries the arm,
            # so saving both cannot collide -- and saving only the first would
            # discard the better model whenever warm wins, which it does.
            if seed == seeds[0]:
                # The training scope belongs in the FILENAME, not only in the
                # metadata. A capped, few-epoch diagnostic and a full-series
                # production candidate are not interchangeable, and the one
                # thing that reliably travels between a console and a
                # deployment is the file name.
                scope = ("full" if not a.train_days
                         else "d%d" % a.train_days)
                tag = "%s_%s_%s_e%d_s%d" % (a.mode, scope, arm, a.epochs, seed)
                os.makedirs(OUTDIR, exist_ok=True)
                mp = os.path.join(OUTDIR, "ev_cnn_lstm_%s.keras" % tag)
                sp = os.path.join(OUTDIR, "ev_scalers_%s.joblib" % tag)
                m.save(mp)
                joblib.dump(scalers, sp)
                # the convention is part of the artefact: a model saved here is
                # only valid when served features built the same way
                with open(os.path.join(OUTDIR,
                                       "ev_%s_metadata.json" % tag), "w") as f:
                    json.dump({"feature_mode": a.mode,
                               "seq_features": SEQ_FEATURES,
                               "lookback": LOOKBACK,
                               "holdout_days": a.hold_days,
                               "train_days": a.train_days or "all",
                               "train_windows": int(len(train[2])),
                               "epochs": a.epochs, "lr": a.lr, "seed": seed,
                               "n_seeds_in_campaign": len(seeds),
                               "scalers": os.path.basename(sp),
                               "holdout_nrmse": round(r["nrmse"], 4),
                               "holdout_persistence": round(r["persistence"], 4),
                               "beats_persistence": bool(
                                   r["nrmse"] < r["persistence"]),
                               "note": "serve features built with mode='%s' or "
                                       "this model is mis-specified. Scalers "
                                       "were re-fitted for this convention and "
                                       "must be deployed with these weights."
                                       % a.mode},
                              f, indent=2)
                print("[retrain] saved %s + %s" % (mp, sp), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    df.to_csv(RESULTS, index=False)

    print("\n=== mean over %d seeds ===" % len(seeds))
    g = df.groupby("arm")[["nrmse", "mae", "bias_kw", "pred_mean", "corr"]].mean()
    print(g.round(4).to_string())
    print("\n  naive persistence on the same holdout: %.4f"
          % df["persistence"].iloc[0])
    print("  deployed model, served features (e19): 1.3890")
    best = g["nrmse"].min()
    print("\n  best refit: %.4f -> %s" % (
        best, "beats persistence" if best < df["persistence"].iloc[0]
        else "STILL WORSE than persistence; the architecture or the feature "
             "set, not the convention, is the binding constraint"))
    print("\n[retrain] wrote %s" % RESULTS)


if __name__ == "__main__":
    main()
