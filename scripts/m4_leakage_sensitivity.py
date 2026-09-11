"""
m4_leakage_sensitivity.py -- does the target leakage in the rolling means change
any conclusion?

THE LEAK. coldstart_transfer/features.engineer_features defines the rolling means
as trailing AND INCLUSIVE:
    roll_1h_mean[t]  = mean(ev[t-3 .. t])
    roll_6h_mean[t]  = mean(ev[t-23 .. t])
    roll_24h_mean[t] = mean(ev[t-95 .. t])
and NEXT_EXO = SEQ_FEATURES[1:] hands the PREDICTED step's features to the model.
So while predicting ev[t] the model receives a quantity containing ev[t]/4. At
inference time that input does not exist, so the deployed pipeline either
computes something different or has a train/serve mismatch.

THE QUESTION. Before deciding whether every campaign must be re-run, measure
whether the leak is actually exploited. It touches 3 of 21 channels, and the
frozen production MinMax scalers compress them; a model that does not use it
would be unaffected by its removal.

METHOD. Build a causal feature frame (rolling means shifted by one step, so
roll[t] = mean(ev[t-4..t-1]) and never contains ev[t]), then compare LIKE FOR
LIKE: a source model trained from scratch on leaky features versus the same
recipe on causal features, evaluated on the same holdouts. The production model
cannot be used here -- it was trained on leaky features, so feeding it causal
ones would confound the leak with a distribution shift in its inputs.

If the causal model is materially worse, the leak was being exploited and the
reported one-step scores are optimistic. If the two match, the leak is inert and
the campaigns stand; document it as a limitation and move on.

Writes Data/results/m4_leakage_sensitivity.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

from coldstart_transfer.features import SEQ_FEATURES
from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.trainer import nrmse

SRC = "Data/lux_source_features.csv"
TGT = "Data/uk_ev_features_full.csv"
SCALERS = "Data/models/ev_scalers.joblib"
OUT = os.environ.get("OUT", "Data/results/m4_leakage_sensitivity.csv")
SRC_DAYS = int(os.environ.get("SRC_DAYS", "150"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
HOLD = int(os.environ.get("HOLD", "14"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
SRC_EPOCHS = int(os.environ.get("SRC_EPOCHS", "12"))

ROLLS = [(4, "roll_1h_mean"), (24, "roll_6h_mean"), (96, "roll_24h_mean")]


def make_causal(df):
    """Recompute the three rolling means WITHOUT the current step.

    Leaky (as deployed):  roll_w[t] = mean(ev[t-w+1 .. t])
    Causal (here):        roll_w[t] = mean(ev[t-w   .. t-1])
    Everything else -- lags, weather, calendar -- is already causal and untouched.
    """
    out = df.copy()
    ev = out["ev_kw"].astype(float)
    for w, col in ROLLS:
        out[col] = ev.shift(1).rolling(w, min_periods=1).mean().fillna(0.0).values
    return out


def windows(df, scalers):
    a, b, c, d = make_windows(df, scalers)
    return a, b, c, d


def train_source(Xs, Xn, y, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    m = build_ev_model(seed=seed)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
    m.fit([Xs, Xn], y, epochs=SRC_EPOCHS, batch_size=128, verbose=0)
    return m


def main():
    scalers = joblib.load(SCALERS)
    src_raw = pd.read_csv(SRC).iloc[-(SRC_DAYS * 96 + LOOKBACK):].reset_index(drop=True)
    tgt_raw = pd.read_csv(TGT).iloc[-(LOOKBACK + HOLD * 96):].reset_index(drop=True)

    # sanity: how different are the two feature versions at all?
    causal_src = make_causal(src_raw)
    for w, col in ROLLS:
        diff = np.abs(src_raw[col].to_numpy(float) - causal_src[col].to_numpy(float))
        print(f"[m4] {col}: mean |leaky - causal| = {diff.mean():.4f} kW "
              f"(signal mean {src_raw.ev_kw.mean():.2f} kW)", flush=True)

    rows = []
    for variant, sdf, tdf in (("leaky", src_raw, tgt_raw),
                              ("causal", causal_src, make_causal(tgt_raw))):
        sXs, sXn, sy, sts = windows(sdf, scalers)
        pool, ho = split_holdout(sXs, sXn, sy, sts, holdout_days=RET_DAYS)
        tXs, tXn, ty, tts = windows(tdf, scalers)

        for seed in SEEDS:
            m = train_source(pool[0], pool[1], pool[2], seed)
            src_n, _, _ = nrmse(m, ho, scalers)
            tgt_n, _, _ = nrmse(m, (tXs, tXn, ty, tts), scalers)
            rows.append(dict(variant=variant, seed=seed,
                             source_holdout_nrmse=round(src_n, 6),
                             target_zeroshot_nrmse=round(tgt_n, 6)))
            print(f"[m4 {variant} s={seed}] source={src_n:.4f} "
                  f"target_zeroshot={tgt_n:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    g = df.groupby("variant")[["source_holdout_nrmse", "target_zeroshot_nrmse"]]
    print("\n" + g.agg(["mean", "std"]).round(4).to_string())

    a = df[df.variant == "leaky"].source_holdout_nrmse.mean()
    b = df[df.variant == "causal"].source_holdout_nrmse.mean()
    print(f"\n[m4] source holdout: leaky {a:.4f} -> causal {b:.4f} "
          f"({100*(b-a)/a:+.1f} %)")
    print("[m4] If the gap is small, the leak is not exploited and the existing "
          "campaigns stand (document as a limitation). If large, they must be "
          "re-run on causal features.")
    print(f"[m4] wrote {OUT}")


if __name__ == "__main__":
    main()
