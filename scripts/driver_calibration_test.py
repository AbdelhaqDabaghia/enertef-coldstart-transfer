"""
driver_calibration_test.py -- is the MinMax "EWC hurts" result caused by the
production model's calibration bias?

THE CONFOUND. Under MinMax, EWC degrades retention (+3.6 to +10 % vs +1.7 % for a
plain fine-tune). Under RevIN, the same variants IMPROVE it (-4.2 to -4.9 %).
But the two arms differ in TWO ways, not one:
  * the representation (MinMax vs RevIN), and
  * the source model -- MinMax warm-starts from the DEPLOYED production model,
    which under-predicts the source holdout by 15.2 kW at corr 0.958
    (scripts/retention_anomaly.py), while RevIN retrains its source model.

A regulariser anchors weights to theta*. If theta* is mis-calibrated, EWC blocks
the re-calibration the model needs -- which would explain the damage without
invoking the representation at all.

THE TEST. Same representation (production MinMax scalers), same data, same
protocol, same conditions. Only the source model changes:
    arm "production" : warm-start from Data/models/ev_cnn_lstm_20260718.keras
    arm "retrained"  : warm-start from a model trained here with the same recipe
                       as the RevIN/clear-sky arms (12 epochs, batch 128, 1e-3)

If EWC stops degrading in the "retrained" arm, the reversal is about calibration
of the deployed model, not representation. If it degrades in both, the
representation is implicated. Either way one hypothesis dies.

Metrics come straight from trainer.run_condition (real production scalers, so
nRMSE is already in kW -- no re-computation needed, unlike the RevIN drivers).

Writes Data/results/calibration_test.csv, notes = "<arm>|<condition>".
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout, LOOKBACK
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.trainer import run_condition, nrmse
from coldstart_transfer.cl_methods import run_cl_method
from coldstart_transfer.logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SCALERS = "Data/models/ev_scalers.joblib"
TARGET = "Data/uk_ev_features_full.csv"
SOURCE = "Data/lux_source_features.csv"
OUT = os.environ.get("OUT", "Data/results/calibration_test.csv")
RETRAINED = os.environ.get("RETRAINED", "Data/models/_minmax_retrained_source.keras")

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
SRC_DAYS = int(os.environ.get("SRC_DAYS", "150"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
SRC_EPOCHS = int(os.environ.get("SRC_EPOCHS", "12"))
ARMS = os.environ.get("ARMS", "production,retrained").split(",")

# same grid as driver_ewc_bench.CONFIGS, plus MAS from the CL suite
CONFIGS = [
    ("B3_warm",           "B3", 0.0,   0.0, "global",   500),
    ("B5_replay",         "B5", 0.0,   1.0, "global",   500),
    ("V1_empFisher_l10",  "V1", 10.0,  0.0, "global",   200),
    ("V1_empFisher_l100", "V1", 100.0, 0.0, "global",   200),
    ("V2_perlayer_l10",   "V2", 10.0,  0.0, "perlayer", 500),
    ("V2_perlayer_l100",  "V2", 100.0, 0.0, "perlayer", 500),
    ("V3_hybrid",         "V3", 10.0,  1.0, "global",   200),
]
CL_METHODS = ["MAS"]

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def main():
    scalers = joblib.load(SCALERS)

    # ---- source: SRC_DAYS span, last RET_DAYS held out for retention ----
    sdf = pd.read_csv(SOURCE).iloc[-(SRC_DAYS * 96 + LOOKBACK):].reset_index(drop=True)
    sXs, sXn, sy, sts = make_windows(sdf, scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=RET_DAYS)

    # ---- target: full train stream + fixed final holdout ----
    tdf = pd.read_csv(TARGET)
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)

    print(f"[cal] source pool={len(source_pool[2])} retention={len(source_holdout[2])} | "
          f"target train={len(target_train[2])} holdout={len(target_holdout[2])}", flush=True)

    # ---- arm B's source model: same recipe as the RevIN / clear-sky arms ----
    if "retrained" in ARMS and not os.path.exists(RETRAINED):
        keras.backend.clear_session()
        keras.utils.set_random_seed(0)
        m = build_ev_model(seed=0)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
        m.fit([source_pool[0], source_pool[1]], source_pool[2],
              epochs=SRC_EPOCHS, batch_size=128, verbose=0)
        os.makedirs(os.path.dirname(RETRAINED), exist_ok=True)
        m.save(RETRAINED)

    # ---- pre-adaptation reference for BOTH arms (the row every table needs) ----
    refs = {}
    for arm, path in (("production", PROD), ("retrained", RETRAINED)):
        if arm not in ARMS:
            continue
        keras.backend.clear_session()
        mm = build_ev_model()
        mm.set_weights(keras.models.load_model(path, compile=False).get_weights())
        r_src, _, _ = nrmse(mm, source_holdout, scalers)
        r_tgt, _, _ = nrmse(mm, target_holdout, scalers)
        refs[arm] = r_src
        print(f"[cal] {arm:11s} pre-adaptation: source={r_src:.4f} target(zero-shot)={r_tgt:.4f}",
              flush=True)

    for arm in ARMS:
        path = PROD if arm == "production" else RETRAINED
        for seed in SEEDS:
            for name, base, lam, beta, mode, mfish in CONFIGS:
                keras.backend.clear_session()
                r = run_condition(base, target_train, target_holdout, scalers, path,
                                  n_days=N_DAYS, lambda_ewc=lam, beta_replay=beta,
                                  seed=seed, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  source_train=source_pool,
                                  source_holdout=source_holdout, m_fisher=mfish,
                                  normalize_fisher=(lam > 0), normalize_mode=mode)
                r.pop("model", None)
                r["notes"] = f"{arm}|{name}"
                log_result(OUT, r)
                print(f"[cal {arm} {name} s={seed}] target={r['target_nrmse']:.4f} "
                      f"retention={r['source_retention_nrmse']:.4f} "
                      f"(ref {refs.get(arm, float('nan')):.4f})", flush=True)

            for meth in CL_METHODS:
                keras.backend.clear_session()
                r = run_cl_method(meth, target_train, target_holdout, scalers, path,
                                  n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                                  batch_size=BATCH, lr=LR, source_train=source_pool,
                                  source_holdout=source_holdout)
                r.pop("model", None)
                r["notes"] = f"{arm}|{meth}"
                log_result(OUT, r)
                print(f"[cal {arm} {meth} s={seed}] target={r['target_nrmse']:.4f} "
                      f"retention={r['source_retention_nrmse']:.4f}", flush=True)

    print(f"[cal] DONE -> {OUT}")


if __name__ == "__main__":
    main()
