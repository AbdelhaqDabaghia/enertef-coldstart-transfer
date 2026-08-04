"""
driver_fisher_control.py -- RQ3 fairness control (answers the CL reviewer's
first objection): the source Fisher's raw scale is ~211x the target Fisher's, so
B1-vs-B4 at a shared lambda is confounded by effective-regularization scale.

Control: NORMALIZE each Fisher by its own mean (scale -> O(1)) AND sweep lambda
for BOTH B1 and B4. Compare B1-best vs B4-best. If B1 stays worse across the
sweep with scales equalized, the transferred Fisher is intrinsically less useful
(not merely miscalibrated).

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_fisher_control
"""
import os, joblib, pandas as pd
from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = "Data/results/rq3_fisher_control.csv"
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "10,100,1000,10000").split(",")]


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (60 + 7) * 96 + 672
    sXs, sXn, sy, sts = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=7)

    print(f"[fisher-control] NORMALIZED Fisher, lambda sweep {LAMBDAS}, "
          f"B1(src) vs B4(tgt), N={N_DAYS}, seeds={SEEDS}")
    for seed in SEEDS:
        for b in ["B1", "B4"]:
            for lam in LAMBDAS:
                r = run_condition(b, target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, lambda_ewc=lam, seed=seed,
                                  epochs=10, batch_size=64, lr=1e-4,
                                  source_train=source_pool, source_holdout=source_holdout,
                                  normalize_fisher=True)
                r.pop("model", None)
                r["notes"] = "fisher_normalized"
                log_result(OUT, r)
                print(f"[{b} lam={lam:g} s={seed}] target={r['target_nrmse']:.4f} "
                      f"retention={r['source_retention_nrmse']} t={r['wall_clock_s']}s", flush=True)
    print(f"[fisher-control] DONE -> {OUT}")


if __name__ == "__main__":
    main()
