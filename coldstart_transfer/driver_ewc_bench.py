"""
driver_ewc_bench.py -- HONEST bench of improved continual-learning variants,
compared fairly (same protocol, 3 seeds, N=30, target + source-retention) against
the plain warm-start (B3) and replay (B5) baselines.

Variants tested:
  V1  EWC, TRUE empirical target Fisher (real labels), globally normalized
  V2  EWC, output-sensitivity Fisher, PER-LAYER normalized (no layer dominates)
  V3  hybrid: empirical-Fisher EWC + replay

Each EWC variant sweeps lambda. We report every run; the conclusion follows the
numbers (if nothing beats B3/B5, that is the honest result). No cherry-picking.

Run (GPU): wsl ... wsl_run.sh -m coldstart_transfer.driver_ewc_bench
"""
import os, joblib, pandas as pd
from .windowing import make_windows, split_holdout
from .trainer import run_condition
from .logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = "Data/results/ewc_bench.csv"
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]

# (name, baseline, lambda, beta, normalize_mode, m_fisher)
CONFIGS = [
    ("B3_warm",        "B3", 0.0,   0.0, "global",   500),
    ("B5_replay",      "B5", 0.0,   1.0, "global",   500),
    ("V1_empFisher_l1",   "V1", 1.0,   0.0, "global",   200),
    ("V1_empFisher_l10",  "V1", 10.0,  0.0, "global",   200),
    ("V1_empFisher_l100", "V1", 100.0, 0.0, "global",   200),
    ("V2_perlayer_l1",    "V2", 1.0,   0.0, "perlayer", 500),
    ("V2_perlayer_l10",   "V2", 10.0,  0.0, "perlayer", 500),
    ("V2_perlayer_l100",  "V2", 100.0, 0.0, "perlayer", 500),
    ("V3_hybrid",         "V3", 10.0,  1.0, "global",   200),
]


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (60 + 7) * 96 + 672
    sXs, sXn, sy, sts = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts, holdout_days=7)

    print(f"[ewc-bench] {len(CONFIGS)} configs x {len(SEEDS)} seeds, N={N_DAYS}")
    for seed in SEEDS:
        for name, b, lam, beta, nmode, mf in CONFIGS:
            r = run_condition(b, target_train, target_holdout, scalers, PROD,
                              n_days=N_DAYS, lambda_ewc=lam, beta_replay=beta, seed=seed,
                              epochs=10, batch_size=64, lr=1e-4,
                              source_train=source_pool, source_holdout=source_holdout,
                              m_fisher=mf, normalize_fisher=(lam > 0), normalize_mode=nmode)
            r.pop("model", None)
            r["notes"] = name
            log_result(OUT, r)
            print(f"[{name} s={seed}] target={r['target_nrmse']:.4f} "
                  f"retention={r['source_retention_nrmse']} t={r['wall_clock_s']}s", flush=True)
    print(f"[ewc-bench] DONE -> {OUT}")


if __name__ == "__main__":
    main()
