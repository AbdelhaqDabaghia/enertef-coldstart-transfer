"""
driver_cl_revin.py -- re-run the EWC bench AND the full CL suite under RevIN.

WHY: ewc_bench.csv and cl_methods_bench.csv are both produced with the frozen
production MinMax scalers -- the worst of the seven representations ablated
(B3 = 0.95 nRMSE vs 0.116 for RevIN). Every conclusion about EWC/replay/MAS/LwF/
DER++/A-GEM is therefore drawn in the regime where the model performs worst, and
where it loses to naive persistence. This closes that gap.

DESIGN: the CL implementations are reused VERBATIM (trainer.run_condition and
cl_methods.run_cl_method). Only the representation changes:
  * windows are built by driver_representation_ablation.raw_windows + RevINRep,
  * the source model is retrained in-representation and saved as .keras, so
    load_production_weights() warm-starts from it exactly as from production,
  * an identity scaler is passed so the trainer's internal metric does not crash;
    those numbers are ignored and target/retention nRMSE are recomputed here in
    kW with RevIN's per-window de-normalisation.

Writes Data/results/cl_revin.csv (notes = condition name, matching the MinMax
tables so stats_report --group notes works identically).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from scripts.driver_representation_ablation import raw_windows, RevINRep, LB, src, tgt
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.trainer import run_condition
from coldstart_transfer.cl_methods import run_cl_method
from coldstart_transfer.logger import log_result

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
SRC_DAYS = int(os.environ.get("SRC_DAYS", "150"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
SRC_EPOCHS = int(os.environ.get("SRC_EPOCHS", "12"))
OUT = os.environ.get("OUT", "Data/results/cl_revin.csv")
SRC_KERAS = os.environ.get("SRC_KERAS", "Data/models/_revin_source.keras")
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]

# same grid as driver_ewc_bench.CONFIGS
EWC_CONFIGS = [
    ("B3_warm",           "B3", 0.0,   0.0, "global",   500),
    ("B5_replay",         "B5", 0.0,   1.0, "global",   500),
    ("V1_empFisher_l1",   "V1", 1.0,   0.0, "global",   200),
    ("V1_empFisher_l10",  "V1", 10.0,  0.0, "global",   200),
    ("V1_empFisher_l100", "V1", 100.0, 0.0, "global",   200),
    ("V2_perlayer_l1",    "V2", 1.0,   0.0, "perlayer", 500),
    ("V2_perlayer_l10",   "V2", 10.0,  0.0, "perlayer", 500),
    ("V2_perlayer_l100",  "V2", 100.0, 0.0, "perlayer", 500),
    ("V3_hybrid",         "V3", 10.0,  1.0, "global",   200),
]
CL_METHODS = ["MAS", "LwF", "DERpp", "AGEM"]

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


class _Identity:
    """Stand-in for a fitted sklearn scaler: the trainer's internal metric needs
    one, but RevIN de-normalises per window so we recompute the real numbers."""
    def inverse_transform(self, x):
        return np.asarray(x)


IDENTITY_SCALERS = {"scaler_y": _Identity()}


def nrmse_kw(model, Xs, Xn, denorm, true_kw):
    pred = model.predict([Xs, Xn], verbose=0).reshape(-1)
    pred_kw = np.clip(denorm(pred), 0, None)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    return rmse / (float(true_kw.mean()) + 1e-9)


def main():
    rep = RevINRep()

    # ---- source: training span + final RET_DAYS retention holdout ----
    src_tail = src.iloc[-(SRC_DAYS * 96 + LB):].reset_index(drop=True)
    sXs, sXn, sy, _ = raw_windows(src_tail)
    cut = RET_DAYS * 96
    sXs_t, sXn_t, sy_t, _ = rep.apply(sXs[:-cut], sXn[:-cut], sy[:-cut])
    sho_Xs, sho_Xn, sho_y, s_denorm = rep.apply(sXs[-cut:], sXn[-cut:], sy[-cut:])
    s_ho_kw = sy[-cut:]

    # ---- target: first N_DAYS for training, final HOLD days for evaluation ----
    thXs, thXn, thy, _ = raw_windows(tgt.iloc[:LB + N_DAYS * 96].reset_index(drop=True))
    tXs_t, tXn_t, ty_t, _ = rep.apply(thXs, thXn, thy)
    hXs, hXn, hy, _ = raw_windows(tgt.iloc[-(LB + HOLD * 96):].reset_index(drop=True))
    tho_Xs, tho_Xn, tho_y, t_denorm = rep.apply(hXs, hXn, hy)
    t_ho_kw = hy

    ts_t = np.arange(len(ty_t))
    target_train = (tXs_t, tXn_t, ty_t, ts_t)
    target_holdout = (tho_Xs, tho_Xn, tho_y, np.arange(len(tho_y)))
    source_train = (sXs_t, sXn_t, sy_t, np.arange(len(sy_t)))
    source_holdout = (sho_Xs, sho_Xn, sho_y, np.arange(len(sho_y)))

    print(f"[cl-revin] source train={len(sy_t)} retention={len(s_ho_kw)} | "
          f"target train={len(ty_t)} holdout={len(t_ho_kw)}", flush=True)

    # ---- source model in-representation, saved so warm-start works unchanged ----
    if not os.path.exists(SRC_KERAS):
        keras.utils.set_random_seed(0)
        srcm = build_ev_model(seed=0)
        srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
        srcm.fit([sXs_t, sXn_t], sy_t, epochs=SRC_EPOCHS, batch_size=128, verbose=0)
        os.makedirs(os.path.dirname(SRC_KERAS), exist_ok=True)
        srcm.save(SRC_KERAS)
        print(f"[cl-revin] source model trained -> {SRC_KERAS} | retention="
              f"{nrmse_kw(srcm, sho_Xs, sho_Xn, s_denorm, s_ho_kw):.4f}", flush=True)
    else:
        print(f"[cl-revin] reusing source model {SRC_KERAS}", flush=True)

    def record(model, tag, seed, extra):
        t = nrmse_kw(model, tho_Xs, tho_Xn, t_denorm, t_ho_kw)
        s = nrmse_kw(model, sho_Xs, sho_Xn, s_denorm, s_ho_kw)
        row = dict(n_days=N_DAYS, seed=seed, target_nrmse=round(t, 6),
                   source_retention_nrmse=round(s, 6), epochs=EPOCHS,
                   batch_size=BATCH, lr=LR, notes=tag, **extra)
        log_result(OUT, row)
        print(f"[cl-revin {tag} s={seed}] target={t:.4f} retention={s:.4f}", flush=True)

    for seed in SEEDS:
        for name, base, lam, beta, mode, mfish in EWC_CONFIGS:
            if ONLY and name not in ONLY:
                continue
            keras.backend.clear_session()
            r = run_condition(base, target_train, target_holdout, IDENTITY_SCALERS,
                              SRC_KERAS, n_days=N_DAYS, lambda_ewc=lam,
                              beta_replay=beta, seed=seed, epochs=EPOCHS,
                              batch_size=BATCH, lr=LR, source_train=source_train,
                              source_holdout=None, m_fisher=mfish,
                              normalize_fisher=(lam > 0), normalize_mode=mode)
            record(r["model"], name, seed,
                   dict(baseline=base, lambda_ewc=lam, beta_replay=beta))

        for meth in CL_METHODS:
            if ONLY and meth not in ONLY:
                continue
            keras.backend.clear_session()
            r = run_cl_method(meth, target_train, target_holdout, IDENTITY_SCALERS,
                              SRC_KERAS, n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                              batch_size=BATCH, lr=LR, source_train=source_train,
                              source_holdout=None)
            record(r["model"], meth, seed, dict(baseline=meth))

    print(f"[cl-revin] DONE -> {OUT}")


if __name__ == "__main__":
    main()
