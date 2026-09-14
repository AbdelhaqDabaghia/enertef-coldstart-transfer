"""
e7_reference_check.py -- what does the leak buy the DEPLOYED model?

No training. Take the production weights unchanged and score them on the same
holdout under both feature conventions. Any difference is attributable to the
features alone, because the model is identical.

This matters more than the adapted comparisons. The deployed service computes
its rolling means causally (ev_features_builder.safe_rolling_mean truncates at
the current index), so if the model was fitted under the inclusive convention
then the number below on the causal row is what the site has been getting in
production -- not the number in any of our papers.

Writes Data/results/corrected_causal_pipeline/e7_reference_check.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, load_production_weights

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = os.environ.get(
    "OUT", "Data/results/corrected_causal_pipeline/e7_reference_check.csv")
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))

ARMS = [("leaky", "Data/lux_source_features.csv", "Data/uk_ev_features_full.csv"),
        ("causal", "Data/lux_source_features_causal.csv",
         "Data/uk_ev_features_causal.csv")]


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (t.mean() + 1e-9))


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, PROD)

    rows = []
    for tag, lux, uk in ARMS:
        # source retention holdout
        sdf = pd.read_csv(lux)
        tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
        a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
        pool, ho = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

        Xs, Xn, y, _ = ho
        t = inverse_target(y, scalers)
        p = np.clip(inverse_target(
            np.clip(m.predict([Xs, Xn], verbose=0).reshape(-1), 0, None), scalers),
            0, None)

        # the one-parameter gauge, fitted on the pool, applied to the holdout
        Xsp, Xnp, yp, _ = pool
        tp = inverse_target(yp, scalers)
        pp = np.clip(inverse_target(
            np.clip(m.predict([Xsp, Xnp], verbose=0).reshape(-1), 0, None), scalers),
            0, None)
        a = float((pp @ tp) / (pp @ pp + 1e-12))

        # naive persistence on the same holdout, for scale
        pers = np.concatenate([[t[0]], t[:-1]])

        # target zero-shot
        tdf = pd.read_csv(uk)
        tXs, tXn, ty, _ = make_windows(tdf, scalers)
        tt = inverse_target(ty, scalers)
        tp_ = np.clip(inverse_target(
            np.clip(m.predict([tXs, tXn], verbose=0).reshape(-1), 0, None), scalers),
            0, None)

        rows.append(dict(
            arm=tag,
            source_nrmse=round(nrmse(p, t), 4),
            source_gauged=round(nrmse(a * p, t), 4),
            gauge_a=round(a, 4),
            persistence_nrmse=round(nrmse(pers, t), 4),
            corr=round(float(np.corrcoef(p, t)[0, 1]), 4),
            bias_kw=round(float((p - t).mean()), 2),
            target_zeroshot=round(nrmse(tp_, tt), 4)))
        print("[e7] %-7s source=%.4f gauged=%.4f a=%.3f | persistence=%.4f "
              "corr=%.3f bias=%+.1f kW | target0=%.4f"
              % (tag, rows[-1]["source_nrmse"], rows[-1]["source_gauged"],
                 a, rows[-1]["persistence_nrmse"], rows[-1]["corr"],
                 rows[-1]["bias_kw"], rows[-1]["target_zeroshot"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n" + df.to_string(index=False))
    l, c = df.iloc[0], df.iloc[1]
    print("\n  causal / leaky on source : %.2fx" % (c.source_nrmse / l.source_nrmse))
    print("  persistence is identical in both arms (it uses no features): %.4f"
          % l.persistence_nrmse)
    print("\n[e7] wrote %s" % OUT)


if __name__ == "__main__":
    main()
