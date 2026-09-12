"""
e15_ecc_mpc_objective.py -- what does forgetting cost in the objective the
deployed controller actually minimises?

e14 priced retention against a spot-price bill. That was the wrong objective:
the Service 1 controller (ECC_MCP_v0.5.ipynb, solve_ecc_mpc) minimises grid
exchange, not cost:

    decision   u[k]                      deviation added to the EV baseline
    grid       Pgrid = (EV_base + u) - PV
    objective  w_import*sum(pos(Pgrid))*dt + w_export*sum(pos(-Pgrid))*dt
               + w_u*||u||^2 + w_ramp*||diff(u)||^2
    subject to |u| <= u_max, 0 <= EV_base+u <= ev_max,
               |diff(u)| <= ramp_max,  and  sum(u) == 0   <- energy is SHIFTED

The last constraint is the one that matters for this study: total energy is
conserved, so only its placement is optimisable. That is exactly the assumption
of the greedy stand-in used earlier -- the deployed controller makes it too.

THE MEASUREMENT. The controller plans on a FORECAST EV baseline. We then settle
the plan against the demand and PV that actually arrived: the realised objective
is computed with the real series and the planned deviation u*. Comparing
forecasters on the planned objective would be meaningless (an optimistic
forecast lowers the plan without lowering anything real).

Conditions: oracle demand, naive persistence, the un-adapted production model,
and the adapted models of each CL strategy. The spread between them is what
forgetting is worth in this system.

Writes Data/results/e15_ecc_mpc_objective.csv.
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
OUT = os.environ.get("OUT", "Data/results/e15_ecc_mpc_objective.csv")

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))

# hyper-parameters transcribed from solve_ecc_mpc (ECC_MCP_v0.5.ipynb)
DT_H = 0.25
U_MAX, EV_MAX, RAMP_MAX = 200.0, 800.0, 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 1.0, 2.0, 0.01, 0.2
H = 96                      # one day of 15-min steps

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def solve_u(pv_fc, ev_fc):
    """The deployed objective, one horizon. Returns the planned deviation u*."""
    u = cp.Variable(H)
    pgrid = (ev_fc + u) - pv_fc
    obj = (W_IMPORT * cp.sum(cp.pos(pgrid)) * DT_H
           + W_EXPORT * cp.sum(cp.pos(-pgrid)) * DT_H
           + W_U * cp.sum_squares(u)
           + W_RAMP * cp.sum_squares(u[1:] - u[:-1]))
    cons = [u >= -U_MAX, u <= U_MAX,
            ev_fc + u >= 0.0, ev_fc + u <= EV_MAX,
            cp.abs(u[1:] - u[:-1]) <= RAMP_MAX,
            cp.sum(u) == 0]
    p = cp.Problem(cp.Minimize(obj), cons)
    try:
        p.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return np.zeros(H)
    return np.zeros(H) if u.value is None else np.asarray(u.value).ravel()


def realised_objective(u, ev_real, pv_real):
    """Settle the planned deviation against what actually happened."""
    pgrid = (ev_real + u) - pv_real
    imp = float(np.sum(np.clip(pgrid, 0, None)) * DT_H)
    exp = float(np.sum(np.clip(-pgrid, 0, None)) * DT_H)
    return W_IMPORT * imp + W_EXPORT * exp, imp, exp


def evaluate(pred_kw, ev_real, pv_real):
    """Plan on the forecast, settle on reality, day by day."""
    tot, imps, exps = [], [], []
    for d in range(len(ev_real) // H):
        s = slice(d * H, (d + 1) * H)
        u = solve_u(pv_real[s], np.clip(pred_kw[s], 0, None))
        o, i, e = realised_objective(u, ev_real[s], pv_real[s])
        tot.append(o); imps.append(i); exps.append(e)
    return float(np.mean(tot)), float(np.mean(imps)), float(np.mean(exps))


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)

    Xs, Xn, y_ho, ts_ho = source_holdout
    ev_real = inverse_target(y_ho, scalers)

    # realised PV over the same timestamps, from the site master file
    site = pd.read_csv(SITE, parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    pv_map = site.set_index("t")["PV_TotalProduction_kW"].astype(float)
    pv_real = pv_map.reindex(pd.to_datetime(ts_ho, utc=True)).ffill().fillna(0.0).to_numpy()
    n = (len(ev_real) // H) * H
    ev_real, pv_real = ev_real[:n], pv_real[:n]
    print(f"[e15] holdout {n} steps = {n//H} days | EV mean {ev_real.mean():.1f} kW | "
          f"PV mean {pv_real.mean():.1f} kW", flush=True)

    def predict(model):
        p = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
        return np.clip(inverse_target(p, scalers), 0, None)[:n]

    rows = []

    def record(tag, seed, pred, extra=None):
        o, i, e = evaluate(pred, ev_real, pv_real)
        rows.append(dict(condition=tag, seed=seed, objective=round(o, 3),
                         import_kwh=round(i, 2), export_kwh=round(e, 2),
                         **(extra or {})))
        print(f"[e15 {tag} s={seed}] objective={o:9.2f}  import={i:8.2f} kWh  "
              f"export={e:8.2f} kWh", flush=True)

    record("oracle_demand", -1, ev_real)
    record("persistence", -1, np.concatenate([[ev_real[0]], ev_real[:-1]]))
    keras.backend.clear_session()
    m0 = build_ev_model(); load_production_weights(m0, PROD)
    record("reference_no_adaptation", -1, predict(m0))

    for seed in SEEDS:
        for name in ["B3_warm", "MAS", "B5_replay"]:
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
            record(name, seed, predict(r["model"]),
                   dict(retention_nrmse=float(r["source_retention_nrmse"])))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== realised ECC-MPC objective (lower is better) ===")
    g = df.groupby("condition").agg(n=("seed", "count"), obj=("objective", "mean"),
                                    sd=("objective", "std"),
                                    nrmse=("retention_nrmse", "mean"))
    base = g.loc["oracle_demand", "obj"]
    g["vs_oracle_%"] = 100 * (g.obj - base) / base
    print(g.round(3).to_string())
    print(f"\n[e15] wrote {OUT}")


if __name__ == "__main__":
    main()
