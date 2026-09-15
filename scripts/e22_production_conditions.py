"""
e22_production_conditions.py -- do the candidates survive a realistic history?

THE GAP THIS CLOSES. e19 and e20 evaluated against the contiguous CSV: fresh
history, every lookback slot real. Production has neither. The ingest runs once
a day and the feature builder fills every unmatched slot with the median, so at
33 h staleness roughly 132 of 672 slots -- the MOST RECENT 132, including
lag_1 -- are a flat constant.

So every accuracy figure this project has published understates the deployed
problem by an unmeasured amount. This measures it.

METHOD. Reproduce the production assembly exactly. For a given staleness S:
    * take the true history
    * discard everything newer than (window_end - S)
    * fill the resulting tail with the median of what remains, as
      ev_features_builder.fill_series_from_history does
then rebuild features from that damaged series and score each candidate.

Staleness levels chosen to match measured operating points rather than a sweep:
    0 h   the clean-CSV assumption behind e19/e20
    9.2 h best case under the OLD 08:55 ingest
   12.4 h expected MEAN after re-pinning to 00:05 (2026-09-15)
   21.1 h measured MEAN under the old schedule
   33.0 h measured WORST case under the old schedule

THE QUESTION. Not "how much worse does everything get" -- that is obvious. It
is whether the causal refit KEEPS its advantage when the recent lookback is
fabricated. If padding destroys the advantage, deploying the new weights
without fixing ingestion buys little, and the ordering of the deployment plan
changes.

Writes Data/results/corrected_causal_pipeline/e22_production_conditions.csv
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

from coldstart_transfer.features import engineer_features, SEQ_FEATURES, WEATHER_VARS
from coldstart_transfer.windowing import (make_windows, split_holdout,
                                          inverse_target, LOOKBACK)
from coldstart_transfer.model import build_ev_model, load_production_weights

MODELS = "Data/models"
LUX = "Data/lux_source_features.csv"
OUT = "Data/results/corrected_causal_pipeline/e22_production_conditions.csv"
RET_DAYS = 7
STEP_MIN = 15

STALENESS_H = [0.0, 9.2, 12.4, 21.1, 33.0]

CANDIDATES = [
    ("production", "ev_cnn_lstm_20260718.keras", "ev_scalers.joblib", "serve"),
    ("causal_s1", "ev_cnn_lstm_causal_full_warm_e20_s1.keras",
     "ev_scalers_causal_full_warm_e20_s1.joblib", "causal"),
    ("causal_s0", "ev_cnn_lstm_causal_full_warm_e20_s0.keras",
     "ev_scalers_causal_full_warm_e20_s0.joblib", "causal"),
]


def stale_and_pad(ev_kw, ts, staleness_h):
    """Reproduce what the deployed builder hands the model.

    Everything newer than (end - staleness) is unavailable; those slots take the
    median of what remains. Applied to the RAW series before feature
    engineering, because that is where production loses the data -- padding
    after feature construction would understate the damage, since the lags and
    rolling means would still have been computed from real values.
    """
    if staleness_h <= 0:
        return ev_kw.copy(), 0
    n_missing = int(round(staleness_h * 60 / STEP_MIN))
    n_missing = min(n_missing, len(ev_kw) - 1)
    out = ev_kw.copy()
    kept = out[:-n_missing]
    pos = kept[kept > 0]
    pad = float(np.median(pos)) if len(pos) else 0.0
    out[-n_missing:] = pad
    return out, n_missing


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (t.mean() + 1e-9))


def main():
    raw_full = pd.read_csv(LUX)[["timestamp", "ev_kw"] + WEATHER_VARS]
    rows = []

    for label, wname, sname, mode in CANDIDATES:
        wpath, spath = os.path.join(MODELS, wname), os.path.join(MODELS, sname)
        if not (os.path.exists(wpath) and os.path.exists(spath)):
            print("[e22] SKIP %s (artefact missing)" % label, flush=True)
            continue
        scalers = joblib.load(spath)
        m = build_ev_model()
        load_production_weights(m, wpath)

        for sh in STALENESS_H:
            raw = raw_full.copy()
            ev_damaged, n_missing = stale_and_pad(
                raw["ev_kw"].to_numpy(float), raw["timestamp"].to_numpy(), sh)
            raw["ev_kw"] = ev_damaged

            feat = engineer_features(raw, mode=mode)
            Xs, Xn, y, ts = make_windows(feat, scalers)
            _, ho = split_holdout(Xs, Xn, y, ts, holdout_days=RET_DAYS)
            hXs, hXn, hy, _ = ho

            # Score against the TRUE target, not the damaged one: the site's
            # real demand arrives regardless of what we could observe. Damaging
            # the target as well would measure how well the model predicts our
            # own padding, which is not a question anyone has.
            feat_true = engineer_features(raw_full, mode=mode)
            _, ho_true = split_holdout(*make_windows(feat_true, scalers),
                                       holdout_days=RET_DAYS)
            t = inverse_target(ho_true[2], scalers)

            p = np.clip(inverse_target(
                np.clip(m.predict([hXs, hXn], verbose=0).reshape(-1), 0, None),
                scalers), 0, None)
            n = min(len(p), len(t))
            p, t = p[:n], t[:n]
            pers = np.concatenate([[t[0]], t[:-1]])

            rows.append(dict(
                candidate=label, feature_mode=mode, staleness_h=sh,
                slots_padded=min(n_missing, LOOKBACK),
                pad_fraction=round(min(n_missing, LOOKBACK) / LOOKBACK, 4),
                nrmse=round(nrmse(p, t), 4),
                mae=round(float(np.mean(np.abs(p - t))), 3),
                bias_kw=round(float((p - t).mean()), 2),
                pred_mean=round(float(p.mean()), 2),
                true_mean=round(float(t.mean()), 2),
                persistence=round(nrmse(pers, t), 4)))
            print("[e22] %-11s stale=%5.1f h  pad=%3d/672  nRMSE=%.4f  "
                  "bias=%+7.2f  pred=%5.1f kW"
                  % (label, sh, rows[-1]["slots_padded"], rows[-1]["nrmse"],
                     rows[-1]["bias_kw"], rows[-1]["pred_mean"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== nRMSE by staleness (persistence %.4f) ==="
          % df.persistence.iloc[0])
    piv = df.pivot(index="staleness_h", columns="candidate", values="nrmse")
    print(piv.to_string())

    if {"causal_s1", "production"} <= set(piv.columns):
        print("\n=== does the causal refit keep its advantage? ===")
        for sh in piv.index:
            adv = piv.loc[sh, "production"] - piv.loc[sh, "causal_s1"]
            print("  stale %5.1f h : causal_s1 better by %+.4f nRMSE%s"
                  % (sh, adv, "" if adv > 0 else "   <-- ADVANTAGE LOST"))
    print("\n[e22] wrote %s" % OUT)


if __name__ == "__main__":
    main()
