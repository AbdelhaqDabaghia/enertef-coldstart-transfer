"""
driver_source.py -- RQ2/RQ3 runs that need the Luxembourg SOURCE features
(lux_source_features.csv): B1 (source Fisher), B5 (replay), and the source-
retention probe on every condition, at a fixed data-rich checkpoint (N=30).

Produces the stability-plasticity tradeoff data: for each condition/seed,
  - target_nrmse   (plasticity, UK holdout)
  - source_retention_nrmse (stability, LU source holdout, same adapted model)

B1 vs B4 (both lambda=100, same everything except Fisher source) isolates the
RQ3 question: does the TRANSFERRED source Fisher carry useful structure vs a
target-re-estimated Fisher.

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_source
"""
from __future__ import annotations
import os, joblib, pandas as pd
from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SCALERS = "Data/models/ev_scalers.joblib"
TARGET = "Data/uk_ev_features_full.csv"
SOURCE = "Data/lux_source_features.csv"
OUT = "Data/results/rq2_rq3_source.csv"

N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
CONDS = os.environ.get("CONDS", "B0,B3,B1,B4,B5").split(",")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
LAMBDA = float(os.environ.get("LAMBDA", "100"))   # B1 and B4 use the SAME lambda
BETA = float(os.environ.get("BETA", "1.0"))       # B5 replay weight
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))


def main():
    scalers = joblib.load(SCALERS)

    # target (UK): fixed 14-day holdout, first N days for training
    tdf = pd.read_csv(TARGET)
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    # source (LU): window only a memory-safe TAIL (POOL_DAYS + RET_DAYS + lookback)
    sdf = pd.read_csv(SOURCE)
    tail_rows = (POOL_DAYS + RET_DAYS) * 96 + 672
    sdf_tail = sdf.iloc[-tail_rows:].reset_index(drop=True)
    sXs, sXn, sy, sts = make_windows(sdf_tail, scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=RET_DAYS)
    print(f"[driver_source] target train={len(target_train[2])} holdout={len(target_holdout[2])} | "
          f"source pool={len(source_pool[2])} retention_holdout={len(source_holdout[2])}")
    print(f"[driver_source] N={N_DAYS} conds={CONDS} seeds={SEEDS} lambda={LAMBDA} beta={BETA}")

    for seed in SEEDS:
        for b in CONDS:
            lam = LAMBDA if b in ("B1", "B4") else 0.0
            beta = BETA if b == "B5" else 0.0
            r = run_condition(b, target_train, target_holdout, scalers, PROD,
                              n_days=N_DAYS, lambda_ewc=lam, beta_replay=beta, seed=seed,
                              epochs=EPOCHS, batch_size=BATCH, lr=LR,
                              source_train=source_pool, source_holdout=source_holdout)
            r.pop("model", None)
            log_result(OUT, r)
            print(f"[{b} s={seed}] target_nRMSE={r['target_nrmse']:.4f} "
                  f"source_retention={r['source_retention_nrmse']} t={r['wall_clock_s']}s", flush=True)
    print(f"[driver_source] DONE -> {OUT}")


if __name__ == "__main__":
    main()
