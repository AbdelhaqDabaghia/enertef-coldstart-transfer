"""
e4_affine_oos.py -- does the affine recalibration generalise out of sample?

retention_anomaly.py fits a*pred+b ON the retention holdout and reports
0.5479 -> 0.3873. Its docstring calls that an upper bound, but the manuscript
quotes it as a measurement. This settles it: fit the affine map on the SOURCE
POOL (the 60 days the model never sees at evaluation) and apply it unchanged to
the 7-day retention holdout.

This is the prerequisite for any claim that retention differences are largely
amplitude re-calibration -- and for the gauge-subspace proposal, whose whole
premise is that the affine correction transfers.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK

POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
OUT = os.environ.get("OUT", "Data/results/e4_affine_oos.csv")


def kw(model, part, scalers):
    Xs, Xn, y, _ = part
    p = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
    return np.clip(inverse_target(p, scalers), 0, None), inverse_target(y, scalers)


def nrmse(p, t):
    return float(np.sqrt(np.mean((np.clip(p, 0, None) - t) ** 2))) / (float(t.mean()) + 1e-9)


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a, b, c, d = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    pool, ho = split_holdout(a, b, c, d, holdout_days=RET_DAYS)

    p_pool, t_pool = kw(m, pool, scalers)
    p_ho, t_ho = kw(m, ho, scalers)

    base_ho = nrmse(p_ho, t_ho)

    # (1) in-sample: fit on the holdout itself -- what retention_anomaly.py did
    A_ho = np.c_[p_ho, np.ones_like(p_ho)]
    c_in, *_ = np.linalg.lstsq(A_ho, t_ho, rcond=None)
    ins = nrmse(A_ho @ c_in, t_ho)

    # (2) OUT OF SAMPLE: fit on the pool, apply to the holdout
    A_pool = np.c_[p_pool, np.ones_like(p_pool)]
    c_oos, *_ = np.linalg.lstsq(A_pool, t_pool, rcond=None)
    oos = nrmse(A_ho @ c_oos, t_ho)

    # (3) scale only, fitted on the pool
    s_pool = float((p_pool @ t_pool) / (p_pool @ p_pool))
    oos_scale = nrmse(s_pool * p_ho, t_ho)

    rows = [dict(variant="as-is", nrmse=round(base_ho, 6), a=1.0, b=0.0),
            dict(variant="affine_in_sample", nrmse=round(ins, 6),
                 a=round(float(c_in[0]), 4), b=round(float(c_in[1]), 4)),
            dict(variant="affine_out_of_sample", nrmse=round(oos, 6),
                 a=round(float(c_oos[0]), 4), b=round(float(c_oos[1]), 4)),
            dict(variant="scale_only_out_of_sample", nrmse=round(oos_scale, 6),
                 a=round(s_pool, 4), b=0.0)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))

    keep = (base_ho - oos) / (base_ho - ins) if base_ho > ins else float("nan")
    print(f"\n  pool n={len(t_pool)}  holdout n={len(t_ho)}")
    print(f"  in-sample gain  : {base_ho:.4f} -> {ins:.4f}")
    print(f"  out-of-sample   : {base_ho:.4f} -> {oos:.4f}")
    print(f"  fraction of the in-sample gain that survives OOS: {100*keep:.1f} %")
    print(f"\n[e4] wrote {OUT}")


if __name__ == "__main__":
    main()
