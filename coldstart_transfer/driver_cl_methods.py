"""
driver_cl_methods.py -- extended CL baseline suite (answers the "missing
baselines" critique): MAS, LwF, DER++, A-GEM, compared to warm-start (B3) and
replay (B5) on EV at N=30 with target + source-retention (3 seeds).

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_cl_methods
"""
import os, joblib, pandas as pd
from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .cl_methods import run_cl_method
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = "Data/results/cl_methods_bench.csv"
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    ttrain, tholdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (60 + 7) * 96 + 672
    sXs, sXn, sy, sts = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    spool, sholdout = split_holdout(sXs, sXn, sy, sts, holdout_days=7)

    print(f"[cl-bench] N={N_DAYS} seeds={SEEDS}")
    for seed in SEEDS:
        # references (existing machinery)
        for b, lam, beta in [("B3", 0.0, 0.0), ("B5", 0.0, 1.0)]:
            r = run_condition(b, ttrain, tholdout, scalers, PROD, n_days=N_DAYS,
                              lambda_ewc=lam, beta_replay=beta, seed=seed, epochs=10,
                              batch_size=64, lr=1e-4, source_train=spool, source_holdout=sholdout)
            r.pop("model", None); log_result(OUT, r)
            print(f"[{b} s={seed}] tgt={r['target_nrmse']:.4f} ret={r['source_retention_nrmse']}", flush=True)
        # new methods
        for method in ["MAS", "LwF", "DERpp", "AGEM"]:
            r = run_cl_method(method, ttrain, tholdout, scalers, PROD, n_days=N_DAYS,
                              seed=seed, epochs=10, batch_size=64, lr=1e-4,
                              source_train=spool, source_holdout=sholdout)
            log_result(OUT, r)
            print(f"[{method} s={seed}] tgt={r['target_nrmse']:.4f} ret={r['source_retention_nrmse']} "
                  f"t={r['wall_clock_s']}s", flush=True)
    print(f"[cl-bench] DONE -> {OUT}")


if __name__ == "__main__":
    main()
