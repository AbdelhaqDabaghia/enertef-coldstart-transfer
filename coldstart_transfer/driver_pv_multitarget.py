"""
driver_pv_multitarget.py -- the multi-target experiment (#7): relate domain
distance to transfer performance across several real PV target sites.

For each OPSD Konstanz PV system, run warm-start (B3) and from-scratch (B0) at a
fixed data-rich budget, logging capacity-normalised error so sites of different
sizes are comparable. A companion analysis relates each site's source->target
domain distance to its transfer performance.

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_pv_multitarget
"""
import os, glob, joblib, numpy as np, pandas as pd
from .pv import build_pv_model, make_pv_windows
from .windowing import split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/pv_lstm_openmeteo.keras"
SCALERS = "Data/models/pv_scalers.joblib"
OUT = "Data/results/pv_multitarget.csv"
N_DAYS = int(os.environ.get("N_DAYS", "90"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
CONDS = os.environ.get("CONDS", "B0,B3").split(",")

TARGETS = sorted(glob.glob("Data/pv_target/konstanz_*features.csv"))


def main():
    scalers = joblib.load(SCALERS)
    print(f"[pv-multi] {len(TARGETS)} targets x {CONDS} x seeds={SEEDS} @N={N_DAYS}")
    for path in TARGETS:
        name = os.path.basename(path).replace("konstanz_", "").replace("_features.csv", "")
        df = pd.read_csv(path)
        cap = float(df["pv_kw"].max())
        Xs, Xn, y, ts = make_pv_windows(df, scalers)
        train, holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)
        for seed in SEEDS:
            for b in CONDS:
                r = run_condition(b, train, holdout, scalers, PROD, n_days=N_DAYS,
                                  seed=seed, epochs=10, batch_size=64, lr=1e-4,
                                  model_builder=build_pv_model)
                r.pop("model", None)
                r["notes"] = name
                r["target_mean_kw"] = cap          # store capacity for nRMSE_cap
                log_result(OUT, r)
                nrmse_cap = r["target_rmse_kw"] / cap
                print(f"[{name} {b} s={seed}] RMSE={r['target_rmse_kw']:.3f}kW "
                      f"cap={cap:.2f} nRMSE_cap={nrmse_cap:.3f}", flush=True)
    print(f"[pv-multi] DONE -> {OUT}")


if __name__ == "__main__":
    main()
