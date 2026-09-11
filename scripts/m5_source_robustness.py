"""
m5_source_robustness.py -- can we predict, before adapting, whether consolidation
will help?

WHAT M3 SHOWED. Same representation, same data, same protocol; only the source
model differs:
    production model : a full fine-tune costs +0.9 % source error; regularisation
                       then HURTS (V1 +3.7 %)
    retrained model  : a full fine-tune costs +88.2 %; regularisation HELPS
                       massively (MAS +11.1 % vs +88.2 %)
So the governing factor is not the representation, and not the source model's
accuracy (the source-quality study found no effect) -- it is how easily the
source model is displaced by fine-tuning. Call it FRAGILITY.

THE CLAIM TO TEST. Fragility is measurable *before* committing to an adaptation
mechanism, from a short unregularised fine-tune, and it predicts whether
consolidation will pay. If it does, an autonomous system has an operational
criterion instead of a design-time guess.

DESIGN. Hold everything fixed and vary only the source model's training budget,
which produces a spread of fragilities:
    for each budget b in EPOCH_LEVELS (plus the deployed production model):
        1. train (or load) the source model
        2. FRAGILITY: run PROBE_STEPS unregularised gradient steps on target data
           and record the source-holdout error after each checkpoint --- this is
           the cheap a-priori measurement
        3. PAYOFF: run the full adaptation with B3_warm and with MAS, and record
           how much retention MAS buys over the plain fine-tune
    then test whether (2) predicts (3).

A positive result gives a decision rule; a negative one kills the mechanism
identified in M3 and must be reported as such.

Writes Data/results/m5_source_robustness.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days, LOOKBACK
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.trainer import run_condition, nrmse
from coldstart_transfer.cl_methods import run_cl_method
from coldstart_transfer.logger import log_result
from coldstart_transfer import ewc as ewc_mod

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SCALERS = "Data/models/ev_scalers.joblib"
TARGET = "Data/uk_ev_features_full.csv"
SOURCE = "Data/lux_source_features.csv"
OUT = os.environ.get("OUT", "Data/results/m5_source_robustness.csv")
CKPT = os.environ.get("CKPT", "Data/models/_m5")

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
SRC_DAYS = int(os.environ.get("SRC_DAYS", "150"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCH_LEVELS = [int(x) for x in os.environ.get("EPOCH_LEVELS", "1,4,12,24").split(",")]
PROBE_STEPS = [int(x) for x in os.environ.get("PROBE_STEPS", "10,25,50,100").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def fragility_probe(model, Xs, Xn, y, holdout, scalers, seed=0):
    """Unregularised fine-tune for a few steps; source error after each stop.

    This is the cheap a-priori measurement: no full adaptation, no second model,
    a few dozen gradient steps.
    """
    opt = keras.optimizers.Adam(LR)
    tvars = model.trainable_variables
    rng = np.random.default_rng(seed)
    n = len(y)
    out, done = {}, 0
    base, _, _ = nrmse(model, holdout, scalers)
    for target_steps in PROBE_STEPS:
        while done < target_steps:
            idx = rng.choice(n, size=min(BATCH, n), replace=False)
            with tf.GradientTape() as tape:
                loss = ewc_mod.huber(tf.constant(y[idx]),
                                     model([tf.constant(Xs[idx]),
                                            tf.constant(Xn[idx])], training=True))
            opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
            done += 1
        cur, _, _ = nrmse(model, holdout, scalers)
        out[target_steps] = (cur - base) / base       # relative degradation
    return base, out


def main():
    scalers = joblib.load(SCALERS)
    sdf = pd.read_csv(SOURCE).iloc[-(SRC_DAYS * 96 + LOOKBACK):].reset_index(drop=True)
    sXs, sXn, sy, sts = make_windows(sdf, scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=RET_DAYS)

    tdf = pd.read_csv(TARGET)
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)
    pXs, pXn, py, _ = first_n_days(*target_train, n_days=N_DAYS)

    os.makedirs(CKPT, exist_ok=True)
    arms = [("production", PROD)]
    for ep in EPOCH_LEVELS:
        path = f"{CKPT}/src_ep{ep}.keras"
        if not os.path.exists(path):
            keras.backend.clear_session()
            keras.utils.set_random_seed(0)
            m = build_ev_model(seed=0)
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
            m.fit([source_pool[0], source_pool[1]], source_pool[2],
                  epochs=ep, batch_size=128, verbose=0)
            m.save(path)
        arms.append((f"ep{ep}", path))

    rows = []
    for arm, path in arms:
        # ---- a-priori fragility probe (cheap) ----
        keras.backend.clear_session()
        m = build_ev_model()
        m.set_weights(keras.models.load_model(path, compile=False).get_weights())
        base, frag = fragility_probe(m, pXs, pXn, py, source_holdout, scalers)
        print(f"[m5] {arm:12s} source_ref={base:.4f} fragility="
              + " ".join(f"{k}st:{100*v:+6.1f}%" for k, v in frag.items()), flush=True)

        # ---- payoff: what regularisation buys after full adaptation ----
        for seed in SEEDS:
            keras.backend.clear_session()
            r0 = run_condition("B3", target_train, target_holdout, scalers, path,
                               n_days=N_DAYS, lambda_ewc=0.0, seed=seed,
                               epochs=EPOCHS, batch_size=BATCH, lr=LR,
                               source_train=source_pool,
                               source_holdout=source_holdout)
            keras.backend.clear_session()
            r1 = run_cl_method("MAS", target_train, target_holdout, scalers, path,
                               n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                               batch_size=BATCH, lr=LR, source_train=source_pool,
                               source_holdout=source_holdout)
            b3 = float(r0["source_retention_nrmse"])
            mas = float(r1["source_retention_nrmse"])
            row = dict(arm=arm, seed=seed, source_ref=round(base, 6),
                       b3_retention=b3, mas_retention=mas,
                       payoff=round(b3 - mas, 6),          # >0 : MAS helps
                       b3_target=float(r0["target_nrmse"]),
                       mas_target=float(r1["target_nrmse"]))
            for k, v in frag.items():
                row[f"frag_{k}"] = round(v, 6)
            rows.append(row)
            print(f"[m5 {arm} s={seed}] B3_ret={b3:.4f} MAS_ret={mas:.4f} "
                  f"payoff={b3-mas:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== does a-priori fragility predict the payoff of consolidation? ===")
    agg = df.groupby("arm").agg(
        source_ref=("source_ref", "first"),
        **{f"frag_{k}": (f"frag_{k}", "first") for k in PROBE_STEPS},
        payoff=("payoff", "mean")).sort_values("payoff")
    print(agg.round(4).to_string())
    for k in PROBE_STEPS:
        c = np.corrcoef(agg[f"frag_{k}"], agg["payoff"])[0, 1]
        print(f"  corr(fragility@{k} steps, payoff) = {c:+.3f}  (n={len(agg)} arms)")
    print(f"\n[m5] wrote {OUT}")


if __name__ == "__main__":
    main()
