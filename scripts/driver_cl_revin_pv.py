"""
driver_cl_revin_pv.py -- the representation-reversal replication on PV.

Under RevIN, the EV CL ordering inverts: all eight regularisation variants beat
replay on retention (dz up to -7.75, unanimous over 10 seeds), whereas under the
deployed MinMax scalers EWC degraded retention and replay dominated
(scripts/driver_cl_revin.py, Data/results/cl_revin.csv).

A reversal seen on one task is not a result. This replicates it on PV
(Luxembourg -> Konstanz), in that task's OWN winning representation: the
clear-sky index (B3 = 0.409 vs 0.645 for RevIN in the PV ablation), reusing the
envelope machinery of driver_representation_ablation_pv.py verbatim.

As in the EV version, the CL implementations themselves are untouched
(trainer.run_condition, cl_methods.run_cl_method) -- only the representation and
the model builder change, so any difference is attributable to representation.

Writes Data/results/cl_revin_pv.csv (notes = condition name, matching the MinMax
tables so stats_report --group notes works identically).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from scripts.driver_representation_ablation_pv import (
    src, tgt, fit_envelope, windows_from, clear_sky_windows, LB)
from coldstart_transfer.pv import build_pv_model
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
OUT = os.environ.get("OUT", "Data/results/cl_revin_pv.csv")
SRC_KERAS = os.environ.get("SRC_KERAS", "Data/models/_csi_pv_source.keras")
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]

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
    def inverse_transform(self, x):
        return np.asarray(x)


IDENTITY_SCALERS = {"scaler_y": _Identity()}


def nrmse_kw(model, Xs, Xn, denorm, true_kw):
    pred = model.predict([Xs, Xn], verbose=0).reshape(-1)
    pred_kw = np.clip(denorm(pred), 0, None)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    return rmse / (float(true_kw.mean()) + 1e-9)


def main():
    # Envelopes fitted on TRAIN portions only (no holdout leak), as in the ablation
    s_fit = np.arange(0, len(src) - 90 * 96)
    t_fit = np.arange(0, len(tgt) - HOLD * 96)
    s_env = fit_envelope(src, s_fit)
    t_env = fit_envelope(tgt, t_fit)

    # ---- source: training span + final RET_DAYS retention holdout ----
    a = max(0, len(src) - (SRC_DAYS * 96 + LB))
    src_tail = src.iloc[a:].reset_index(drop=True)
    sXs, sXn, sy, s_es, s_ep = windows_from(src_tail, s_env[a:])
    cut = RET_DAYS * 96
    sXs_t, sXn_t, sy_t, _ = clear_sky_windows(sXs[:-cut], sXn[:-cut], sy[:-cut],
                                              s_es[:-cut], s_ep[:-cut])
    sho_Xs, sho_Xn, sho_y, s_denorm = clear_sky_windows(
        sXs[-cut:], sXn[-cut:], sy[-cut:], s_es[-cut:], s_ep[-cut:])
    s_ho_kw = sy[-cut:]

    # ---- target: first N_DAYS train, final HOLD days holdout ----
    th = tgt.iloc[:LB + N_DAYS * 96].reset_index(drop=True)
    thXs, thXn, thy, th_es, th_ep = windows_from(th, t_env[:LB + N_DAYS * 96])
    tXs_t, tXn_t, ty_t, _ = clear_sky_windows(thXs, thXn, thy, th_es, th_ep)
    b = len(tgt) - (LB + HOLD * 96)
    hd = tgt.iloc[b:].reset_index(drop=True)
    hXs, hXn, hy, h_es, h_ep = windows_from(hd, t_env[b:])
    tho_Xs, tho_Xn, tho_y, t_denorm = clear_sky_windows(hXs, hXn, hy, h_es, h_ep)
    t_ho_kw = hy

    target_train = (tXs_t, tXn_t, ty_t, np.arange(len(ty_t)))
    target_holdout = (tho_Xs, tho_Xn, tho_y, np.arange(len(tho_y)))
    source_train = (sXs_t, sXn_t, sy_t, np.arange(len(sy_t)))

    print(f"[pv-csi] source train={len(sy_t)} retention={len(s_ho_kw)} | "
          f"target train={len(ty_t)} holdout={len(t_ho_kw)}", flush=True)

    if not os.path.exists(SRC_KERAS):
        keras.utils.set_random_seed(0)
        srcm = build_pv_model(seed=0)
        srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
        srcm.fit([sXs_t, sXn_t], sy_t, epochs=SRC_EPOCHS, batch_size=128, verbose=0)
        os.makedirs(os.path.dirname(SRC_KERAS), exist_ok=True)
        srcm.save(SRC_KERAS)
        print(f"[pv-csi] source model -> {SRC_KERAS} | pre-adaptation retention="
              f"{nrmse_kw(srcm, sho_Xs, sho_Xn, s_denorm, s_ho_kw):.4f}", flush=True)
    else:
        print(f"[pv-csi] reusing source model {SRC_KERAS}", flush=True)

    def record(model, tag, seed, extra):
        t = nrmse_kw(model, tho_Xs, tho_Xn, t_denorm, t_ho_kw)
        s = nrmse_kw(model, sho_Xs, sho_Xn, s_denorm, s_ho_kw)
        log_result(OUT, dict(n_days=N_DAYS, seed=seed, target_nrmse=round(t, 6),
                             source_retention_nrmse=round(s, 6), epochs=EPOCHS,
                             batch_size=BATCH, lr=LR, notes=tag, **extra))
        print(f"[pv-csi {tag} s={seed}] target={t:.4f} retention={s:.4f}", flush=True)

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
                              normalize_fisher=(lam > 0), normalize_mode=mode,
                              model_builder=build_pv_model)
            record(r["model"], name, seed,
                   dict(baseline=base, lambda_ewc=lam, beta_replay=beta))

        for meth in CL_METHODS:
            if ONLY and meth not in ONLY:
                continue
            keras.backend.clear_session()
            r = run_cl_method(meth, target_train, target_holdout, IDENTITY_SCALERS,
                              SRC_KERAS, n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                              batch_size=BATCH, lr=LR, source_train=source_train,
                              source_holdout=None, model_builder=build_pv_model)
            record(r["model"], meth, seed, dict(baseline=meth))

    print(f"[pv-csi] DONE -> {OUT}")


if __name__ == "__main__":
    main()
