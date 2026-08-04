"""
driver_pv_source.py -- PV RQ2/RQ3: B1 (source Fisher), B4 (target Fisher),
B5 (replay), B3/B0, with the source-retention probe, at N=30. Mirrors
driver_source.py but with the PV model/windowing and the Luxembourg PV source.

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_pv_source
"""
import os, joblib, pandas as pd
from .pv import build_pv_model, make_pv_windows, PV_LOOKBACK
from .windowing import split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/pv_lstm_openmeteo.keras"
SCALERS = "Data/models/pv_scalers.joblib"
TARGET = "Data/pv_target/konstanz_pv_features.csv"
SOURCE = "Data/pv_target/lux_pv_source_features.csv"
OUT = "Data/results/pv_rq2_rq3_source.csv"

N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
CONDS = os.environ.get("CONDS", "B0,B3,B1,B4,B5").split(",")
LAMBDA = float(os.environ.get("LAMBDA", "100"))
BETA = float(os.environ.get("BETA", "1.0"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))


def main():
    scalers = joblib.load(SCALERS)
    tdf = pd.read_csv(TARGET)
    Xs, Xn, y, ts = make_pv_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv(SOURCE)
    tail = (POOL_DAYS + RET_DAYS) * 96 + PV_LOOKBACK
    sXs, sXn, sy, sts = make_pv_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=RET_DAYS)
    print(f"[pv-src] target train={len(target_train[2])} holdout={len(target_holdout[2])} | "
          f"source pool={len(source_pool[2])} retention={len(source_holdout[2])}")

    for seed in SEEDS:
        for b in CONDS:
            lam = LAMBDA if b in ("B1", "B4") else 0.0
            beta = BETA if b == "B5" else 0.0
            r = run_condition(b, target_train, target_holdout, scalers, PROD,
                              n_days=N_DAYS, lambda_ewc=lam, beta_replay=beta, seed=seed,
                              epochs=10, batch_size=64, lr=1e-4,
                              source_train=source_pool, source_holdout=source_holdout,
                              model_builder=build_pv_model)
            r.pop("model", None); r["notes"] = "PV"
            log_result(OUT, r)
            print(f"[PV {b} s={seed}] target={r['target_nrmse']:.4f} "
                  f"retention={r['source_retention_nrmse']} t={r['wall_clock_s']}s", flush=True)
    print(f"[pv-src] DONE -> {OUT}")


if __name__ == "__main__":
    main()
