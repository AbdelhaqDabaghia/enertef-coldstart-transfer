"""
driver.py -- RQ1 cold-start grid runner (GPU).

Runs the conditions that need ONLY the target (UK) + the transferred model:
    B0 (from-scratch), B3 (warm-start no EWC), B4 (warm + target-Fisher EWC)
over N-day checkpoints x seeds, logging every run to a single results CSV for
the RQ1 nRMSE-vs-N / data-to-threshold analysis.

B1 (source Fisher) and B5 (replay) + the source-retention probe are added once
the Luxembourg source features are exported (export_lux_source.py).

Run (GPU/WSL):
    wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/dev/service1/scripts/wsl_run.sh \
        -m coldstart_transfer.driver
"""
from __future__ import annotations
import os
import joblib
import pandas as pd

from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SCALERS = "Data/models/ev_scalers.joblib"
TARGET = "Data/uk_ev_features_full.csv"
OUT = "Data/results/rq1_coldstart.csv"

N_DAYS = [int(x) for x in os.environ.get("N_DAYS", "1,7,30,60,90").split(",")]
BASELINES = os.environ.get("BASELINES", "B0,B3,B4").split(",")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
LAMBDA_B4 = float(os.environ.get("LAMBDA_B4", "100"))
# data-to-threshold: nRMSE gate. Defensible default; document choice in analysis.
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))


def main():
    scalers = joblib.load(SCALERS)
    df = pd.read_csv(TARGET)
    Xs, Xn, y, ts = make_windows(df, scalers)
    train, holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)
    print(f"[driver] target windows={len(y)} train={len(train[2])} holdout={len(holdout[2])}")
    print(f"[driver] grid: {BASELINES} x N={N_DAYS} x seeds={SEEDS} "
          f"= {len(BASELINES)*len(N_DAYS)*len(SEEDS)} runs, epochs={EPOCHS}")

    for seed in SEEDS:
        for n in N_DAYS:
            for b in BASELINES:
                lam = LAMBDA_B4 if b == "B4" else 0.0
                r = run_condition(b, train, holdout, scalers, PROD,
                                  n_days=n, lambda_ewc=lam, seed=seed,
                                  epochs=EPOCHS, batch_size=BATCH, lr=LR)
                r.pop("model", None)
                r["threshold"] = THRESHOLD
                r["below_threshold"] = int(r["target_nrmse"] < THRESHOLD)
                log_result(OUT, r)
                print(f"[{b} N={n} s={seed}] nRMSE={r['target_nrmse']:.4f} "
                      f"below={r['below_threshold']} t={r['wall_clock_s']}s", flush=True)
    print(f"[driver] DONE -> {OUT}")


if __name__ == "__main__":
    main()
