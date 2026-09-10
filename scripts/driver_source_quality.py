"""
driver_source_quality.py -- does consolidation help only when there is knowledge
to preserve?

Three configurations so far suggest the governing variable is not the
representation but the QUALITY of the source model:

    EV MinMax     source 0.5479   EWC/MAS hurt   (+3.6 to +10 %)
    EV RevIN      source 0.2032   EWC/MAS help   (-4.2 to -4.9 %)
    PV clear-sky  source 0.6086   EWC/MAS hurt   (+15 to +20 %)

Three points are not a law. This turns the observation into a controlled
experiment: hold task, representation, data, architecture and protocol fixed, and
vary ONLY the source model's quality by training it for different numbers of
epochs. If the hypothesis holds, the retention advantage of consolidation over a
plain fine-tune should grow with source quality, and cross zero somewhere.

Everything else is reused verbatim (RevIN windows from the ablation,
trainer.run_condition, cl_methods.run_cl_method), so the only manipulated factor
is source quality.

Writes Data/results/source_quality.csv with one row per
(src_epochs, condition, seed), carrying the measured pre-adaptation source error
so the relationship can be regressed directly.
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
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EPOCH_LEVELS = [int(x) for x in os.environ.get("EPOCH_LEVELS", "1,2,4,8,12,20").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
OUT = os.environ.get("OUT", "Data/results/source_quality.csv")
CKPT_DIR = os.environ.get("CKPT_DIR", "Data/models/_srcq")

# one representative per family, to keep the grid affordable
CONDS = [("B3_warm",          "B3", 0.0,  0.0, "global",   500),
         ("V1_empFisher_l10", "V1", 10.0, 0.0, "global",   200),
         ("V2_perlayer_l100", "V2", 100.0, 0.0, "perlayer", 500),
         ("B5_replay",        "B5", 0.0,  1.0, "global",   500)]
CL_METHODS = ["MAS"]

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
    return float(np.sqrt(np.mean((pred_kw - true_kw) ** 2))) / (float(true_kw.mean()) + 1e-9)


def main():
    rep = RevINRep()
    src_tail = src.iloc[-(SRC_DAYS * 96 + LB):].reset_index(drop=True)
    sXs, sXn, sy, _ = raw_windows(src_tail)
    cut = RET_DAYS * 96
    sXs_t, sXn_t, sy_t, _ = rep.apply(sXs[:-cut], sXn[:-cut], sy[:-cut])
    sho_Xs, sho_Xn, sho_y, s_denorm = rep.apply(sXs[-cut:], sXn[-cut:], sy[-cut:])
    s_ho_kw = sy[-cut:]

    thXs, thXn, thy, _ = raw_windows(tgt.iloc[:LB + N_DAYS * 96].reset_index(drop=True))
    tXs_t, tXn_t, ty_t, _ = rep.apply(thXs, thXn, thy)
    hXs, hXn, hy, _ = raw_windows(tgt.iloc[-(LB + HOLD * 96):].reset_index(drop=True))
    tho_Xs, tho_Xn, tho_y, t_denorm = rep.apply(hXs, hXn, hy)
    t_ho_kw = hy

    target_train = (tXs_t, tXn_t, ty_t, np.arange(len(ty_t)))
    target_holdout = (tho_Xs, tho_Xn, tho_y, np.arange(len(tho_y)))
    source_train = (sXs_t, sXn_t, sy_t, np.arange(len(sy_t)))
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[srcq] levels={EPOCH_LEVELS} seeds={SEEDS} "
          f"conds={[c[0] for c in CONDS] + CL_METHODS}", flush=True)

    for ep in EPOCH_LEVELS:
        ckpt = f"{CKPT_DIR}/src_ep{ep}.keras"
        if not os.path.exists(ckpt):
            keras.backend.clear_session()
            keras.utils.set_random_seed(0)
            m = build_ev_model(seed=0)
            m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
            m.fit([sXs_t, sXn_t], sy_t, epochs=ep, batch_size=128, verbose=0)
            m.save(ckpt)
        else:
            m = keras.models.load_model(ckpt, compile=False)
        q = nrmse_kw(m, sho_Xs, sho_Xn, s_denorm, s_ho_kw)
        print(f"[srcq] === src_epochs={ep}: pre-adaptation source nRMSE={q:.4f}", flush=True)

        def record(model, tag, seed, extra):
            t = nrmse_kw(model, tho_Xs, tho_Xn, t_denorm, t_ho_kw)
            s = nrmse_kw(model, sho_Xs, sho_Xn, s_denorm, s_ho_kw)
            log_result(OUT, dict(n_days=N_DAYS, seed=seed,
                                 target_nrmse=round(t, 6),
                                 source_retention_nrmse=round(s, 6),
                                 epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                 notes=f"ep{ep}|{tag}|q{q:.4f}", **extra))
            print(f"[srcq ep={ep} {tag} s={seed}] target={t:.4f} retention={s:.4f} "
                  f"(src_q={q:.4f})", flush=True)

        for seed in SEEDS:
            for name, base, lam, beta, mode, mfish in CONDS:
                keras.backend.clear_session()
                r = run_condition(base, target_train, target_holdout,
                                  IDENTITY_SCALERS, ckpt, n_days=N_DAYS,
                                  lambda_ewc=lam, beta_replay=beta, seed=seed,
                                  epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  source_train=source_train, source_holdout=None,
                                  m_fisher=mfish, normalize_fisher=(lam > 0),
                                  normalize_mode=mode)
                record(r["model"], name, seed,
                       dict(baseline=base, lambda_ewc=lam, beta_replay=beta))
            for meth in CL_METHODS:
                keras.backend.clear_session()
                r = run_cl_method(meth, target_train, target_holdout,
                                  IDENTITY_SCALERS, ckpt, n_days=N_DAYS, seed=seed,
                                  epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  source_train=source_train, source_holdout=None)
                record(r["model"], meth, seed, dict(baseline=meth))

    print(f"[srcq] DONE -> {OUT}")


if __name__ == "__main__":
    main()
