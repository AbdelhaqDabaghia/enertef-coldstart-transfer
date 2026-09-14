"""
e16_selection_oracle.py -- is there anything for an orchestrator to win?

THE PROPOSAL. Route between forecasting services according to their downstream
economic impact rather than their forecast accuracy. Motivated by e15, which
shows the ordering inverts: replay is best on retention nRMSE (0.418) and worst
on the deployed controller's objective (1197.0), MAS the other way round.

THE PRIOR. M1 already tested per-cycle selection among four CL strategies and
found that greedy selection does NOT beat a well-chosen fixed strategy -- even an
oracle that cheats. Before building an orchestrator, measure the ceiling.

THE TEST. Per (seed, day), compute the realised controller objective for each
candidate, then compare:
    fixed A / fixed B / fixed C     -- commit to one service
    oracle                          -- the best candidate each day, known after
                                       the fact: the CEILING any orchestrator
                                       can approach but never exceed
    by-MAE-today                    -- pick the lowest forecast error of the day
                                       (also unrealisable: needs the outcome)
    by-MAE-yesterday                -- pick whoever had the lowest error the day
                                       before: REALISABLE online, the honest
                                       orchestrator baseline

If the oracle does not beat the best fixed strategy by a clear margin, no
orchestrator can, because a real one has strictly less information. That kills
the idea for one day of work instead of a prototype's worth.

Writes Data/results/e16_selection_oracle.csv (per seed, day, condition).
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import cvxpy as cp
import tensorflow as tf
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.trainer import run_condition
from coldstart_transfer.cl_methods import run_cl_method

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SITE = "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv"
OUT = os.environ.get("OUT", "Data/results/e16_selection_oracle.csv")

# E7: both feature pipelines are runnable without editing this file.
# Default = the historical convention (rolling means include y_t);
# set LUX_CSV / UK_CSV to the *_causal.csv files for the causal pipeline.
LUX_CSV = os.environ.get("LUX_CSV", "Data/lux_source_features.csv")
UK_CSV = os.environ.get("UK_CSV", "Data/uk_ev_features_full.csv")

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EPOCHS, BATCH, LR = 10, 64, 1e-4
DT_H, H = 0.25, 96
U_MAX, EV_MAX, RAMP_MAX = 200.0, 800.0, 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 1.0, 2.0, 0.01, 0.2
CANDIDATES = ["B3_warm", "MAS", "B5_replay"]

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def solve_u(pv, ev_fc):
    u = cp.Variable(H)
    pgrid = (ev_fc + u) - pv
    obj = (W_IMPORT * cp.sum(cp.pos(pgrid)) * DT_H
           + W_EXPORT * cp.sum(cp.pos(-pgrid)) * DT_H
           + W_U * cp.sum_squares(u) + W_RAMP * cp.sum_squares(u[1:] - u[:-1]))
    cons = [u >= -U_MAX, u <= U_MAX, ev_fc + u >= 0.0, ev_fc + u <= EV_MAX,
            cp.abs(u[1:] - u[:-1]) <= RAMP_MAX, cp.sum(u) == 0]
    try:
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return np.zeros(H)
    return np.zeros(H) if u.value is None else np.asarray(u.value).ravel()


def day_objective(pred, ev_real, pv_real):
    u = solve_u(pv_real, np.clip(pred, 0, None))
    pg = (ev_real + u) - pv_real
    return (W_IMPORT * float(np.sum(np.clip(pg, 0, None)) * DT_H)
            + W_EXPORT * float(np.sum(np.clip(-pg, 0, None)) * DT_H))


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sdf = pd.read_csv(LUX_CSV)
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

    tdf = pd.read_csv(UK_CSV)
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)

    Xs, Xn, y_ho, ts_ho = source_holdout
    ev_real = inverse_target(y_ho, scalers)
    site = pd.read_csv(SITE, parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    pv_real = (site.set_index("t")["PV_TotalProduction_kW"].astype(float)
               .reindex(pd.to_datetime(ts_ho, utc=True)).ffill().fillna(0.0).to_numpy())
    n = (len(ev_real) // H) * H
    ev_real, pv_real = ev_real[:n], pv_real[:n]
    ndays = n // H
    print(f"[e16] {ndays} days x {len(SEEDS)} seeds = {ndays*len(SEEDS)} decisions",
          flush=True)

    rows = []
    for seed in SEEDS:
        for name in CANDIDATES:
            keras.backend.clear_session()
            if name == "MAS":
                r = run_cl_method("MAS", target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, seed=seed, epochs=EPOCHS,
                                  batch_size=BATCH, lr=LR, source_train=source_pool,
                                  source_holdout=source_holdout)
            else:
                base = "B5" if name == "B5_replay" else "B3"
                r = run_condition(base, target_train, target_holdout, scalers, PROD,
                                  n_days=N_DAYS, lambda_ewc=0.0,
                                  beta_replay=1.0 if base == "B5" else 0.0,
                                  seed=seed, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                                  source_train=source_pool, source_holdout=source_holdout)
            p = np.clip(r["model"].predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
            pred = np.clip(inverse_target(p, scalers), 0, None)[:n]
            for d in range(ndays):
                s = slice(d * H, (d + 1) * H)
                rows.append(dict(seed=seed, day=d, condition=name,
                                 objective=round(day_objective(pred[s], ev_real[s],
                                                               pv_real[s]), 4),
                                 mae=round(float(np.mean(np.abs(pred[s] - ev_real[s]))), 4)))
            print(f"[e16 {name} s={seed}] done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    obj = df.pivot_table(index=["seed", "day"], columns="condition", values="objective")
    mae = df.pivot_table(index=["seed", "day"], columns="condition", values="mae")

    print("\n=== fixed strategies (mean realised objective) ===")
    for c in CANDIDATES:
        print(f"  always {c:12s} {obj[c].mean():9.3f}")
    best_fixed = min(CANDIDATES, key=lambda c: obj[c].mean())

    oracle = obj.min(axis=1).mean()
    by_mae = obj.values[np.arange(len(obj)), [obj.columns.get_loc(c)
                                              for c in mae.idxmin(axis=1)]].mean()
    # realisable: pick whoever was best (by MAE) the previous day, per seed
    picks = []
    for (seed, day) in obj.index:
        prev = (seed, day - 1)
        c = mae.loc[prev].idxmin() if prev in mae.index else best_fixed
        picks.append(obj.loc[(seed, day), c])
    lagged = float(np.mean(picks))

    print("\n=== selection strategies ===")
    print(f"  best fixed ({best_fixed:12s}) {obj[best_fixed].mean():9.3f}")
    print(f"  by-MAE-today (unrealisable)  {by_mae:9.3f}")
    print(f"  by-MAE-yesterday (realisable){lagged:9.3f}")
    print(f"  ORACLE (ceiling)             {oracle:9.3f}")
    print(f"\n  ceiling over best fixed: {oracle - obj[best_fixed].mean():+.3f} "
          f"({100*(oracle - obj[best_fixed].mean())/obj[best_fixed].mean():+.2f} %)")
    print("  If that margin is not clearly negative, no orchestrator can win.")
    print(f"\n[e16] wrote {OUT}")


if __name__ == "__main__":
    main()
