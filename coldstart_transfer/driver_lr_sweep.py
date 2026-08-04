"""
driver_lr_sweep.py -- the simple anti-forgetting baseline a reviewer expects:
control the ADAPTATION STRENGTH (learning rate x epochs) of a plain warm-start
fine-tune (B3, no EWC, no replay). Lower LR / fewer epochs => the model stays
closer to the source => less forgetting (better source-retention), at some target
cost. This tests whether a trivial knob rivals EWC/replay on the tradeoff.

N=30, 3 seeds, target + source-retention.  Run (GPU):
    wsl ... wsl_run.sh -m coldstart_transfer.driver_lr_sweep
"""
import os, joblib, pandas as pd
from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = "Data/results/lr_sweep.csv"
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LRS = [float(x) for x in os.environ.get("LRS", "1e-4,3e-5,1e-5").split(",")]
EPOCHS = [int(x) for x in os.environ.get("EPOCHS", "3,10").split(",")]


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (60 + 7) * 96 + 672
    sXs, sXn, sy, sts = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    _, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=7)

    print(f"[lr-sweep] B3 warm-start, LR={LRS} x epochs={EPOCHS}, N={N_DAYS}, seeds={SEEDS}")
    for seed in SEEDS:
        for lr in LRS:
            for ep in EPOCHS:
                r = run_condition("B3", target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, seed=seed, epochs=ep, batch_size=64, lr=lr,
                                  source_holdout=source_holdout)
                r.pop("model", None)
                r["notes"] = f"lr{lr:g}_ep{ep}"
                log_result(OUT, r)
                print(f"[lr={lr:g} ep={ep} s={seed}] target={r['target_nrmse']:.4f} "
                      f"retention={r['source_retention_nrmse']} t={r['wall_clock_s']}s", flush=True)
    print(f"[lr-sweep] DONE -> {OUT}")


if __name__ == "__main__":
    main()
