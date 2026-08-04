"""
driver_pv.py -- RQ1 cold-start grid for the PV task (Luxembourg -> Konstanz),
the second-task replication. Reuses the exact protocol/trainer with the PV model
builder and PV windowing. B0/B3/B4 need only the target + the transferred PV model.

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_pv
"""
import os, joblib, pandas as pd
from .pv import build_pv_model, make_pv_windows
from .windowing import split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/pv_lstm_openmeteo.keras"
SCALERS = "Data/models/pv_scalers.joblib"
TARGET = "Data/pv_target/konstanz_pv_features.csv"
OUT = "Data/results/pv_rq1_coldstart.csv"

N_DAYS = [int(x) for x in os.environ.get("N_DAYS", "1,7,30,60,90").split(",")]
BASELINES = os.environ.get("BASELINES", "B0,B3,B4").split(",")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
LAMBDA_B4 = float(os.environ.get("LAMBDA_B4", "100"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))


def main():
    scalers = joblib.load(SCALERS)
    df = pd.read_csv(TARGET)
    Xs, Xn, y, ts = make_pv_windows(df, scalers)
    train, holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)
    print(f"[pv-driver] windows={len(y)} train={len(train[2])} holdout={len(holdout[2])} | "
          f"grid {BASELINES} x N={N_DAYS} x seeds={SEEDS}")
    for seed in SEEDS:
        for n in N_DAYS:
            for b in BASELINES:
                lam = LAMBDA_B4 if b == "B4" else 0.0
                r = run_condition(b, train, holdout, scalers, PROD,
                                  n_days=n, lambda_ewc=lam, seed=seed,
                                  epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  model_builder=build_pv_model)
                r.pop("model", None)
                r["threshold"] = THRESHOLD
                r["below_threshold"] = int(r["target_nrmse"] < THRESHOLD)
                r["notes"] = "PV"
                log_result(OUT, r)
                print(f"[PV {b} N={n} s={seed}] nRMSE={r['target_nrmse']:.4f} "
                      f"t={r['wall_clock_s']}s", flush=True)
    print(f"[pv-driver] DONE -> {OUT}")


if __name__ == "__main__":
    main()
