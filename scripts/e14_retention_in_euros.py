"""
e14_retention_in_euros.py -- what does forgetting cost?

THE QUESTION. The continual-learning literature reports forgetting in forecast
error. Our sequential study reports backward transfer of +7 to +10 kW, and the
transfer study reports retention gaps of 0.13 nRMSE between mechanisms. Those
look large. But the closed-loop study bounds the total value of forecasting on
this site at 1.74 EUR/day (1.55 % of the bill): the gap between naive
persistence and perfect foresight.

So: convert the retention gap into euros. If the mechanisms that structure the
CL literature differ by cents, that is a result -- it says the quantity the
field optimises is not the quantity that matters here.

SCOPE. Only source retention can be priced: the MPC runs on the Luxembourg site,
which is exactly where the retention holdout lives. Target error on the UK site
has no controller and cannot be converted.

METHOD. For each adapted model, predict EV demand over the 7-day source
retention holdout, schedule against those predictions, and settle the plan
against the demand that actually arrived (realised cost, not planned cost --
an optimistic forecast lowers the planned bill without lowering the real one).
Compare conditions, plus two anchors: the un-adapted production model, and
naive persistence.

Reuses the settlement rule of mpc_forecast_value.py verbatim.

Writes Data/results/e14_retention_in_euros.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.trainer import run_condition
from coldstart_transfer.cl_methods import run_cl_method

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
OUT = os.environ.get("OUT", "Data/results/e14_retention_in_euros.csv")
PRICE_CSV = os.environ.get("PRICE_CSV", "/mnt/c/dev/MPC_RL/BESS_MPC_PROJECT/prices_da_real.csv")

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
STEPS_PER_DAY, DT_H = 96, 0.25
CHARGER_CAP_KW = float(os.environ.get("CHARGER_CAP_KW", "200"))

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def greedy_plan(price, demand_kwh):
    """Place the day's charging demand in the cheapest quarters (same rule as
    mpc_forecast_value.greedy_plan, PV term dropped: here the forecast under
    test is the DEMAND, and PV is taken as realised)."""
    order = np.argsort(price)
    plan = np.zeros(STEPS_PER_DAY)
    remaining = demand_kwh
    for i in order:
        if remaining <= 0:
            break
        take = min(CHARGER_CAP_KW * DT_H, remaining)
        plan[i] = take / DT_H
        remaining -= take
    return plan


def realised_cost(plan, price, ev_real):
    """Settle a plan sized on a FORECAST demand against the demand that arrived.
    The real demand must be served: whatever was not planned is charged as it
    arrives, at the prevailing price."""
    e_plan = float(np.sum(plan) * DT_H)
    e_real = float(np.sum(ev_real) * DT_H)
    served = plan * min(1.0, e_real / e_plan) if e_plan > 0 else np.zeros(STEPS_PER_DAY)
    short = max(0.0, e_real - float(np.sum(served) * DT_H))
    if short > 0 and ev_real.sum() > 0:
        served = served + ev_real * (short / (float(np.sum(ev_real) * DT_H)))
    return float(np.sum(price * served * DT_H))


def price_days():
    p = pd.read_csv(PRICE_CSV)
    col = "price_eur_kwh" if "price_eur_kwh" in p.columns else "price"
    v = p[col].astype(float).to_numpy()
    if v.mean() > 5:
        v = v / 1000.0
    nd = len(v) // STEPS_PER_DAY
    return v[:nd * STEPS_PER_DAY].reshape(nd, STEPS_PER_DAY)


def cost_of(pred_kw, true_kw, prices):
    """Daily realised cost when the schedule is sized on pred_kw."""
    out = []
    nd = len(true_kw) // STEPS_PER_DAY
    for d in range(nd):
        s = slice(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        pr = prices[d % len(prices)]
        dmd = float(np.sum(np.clip(pred_kw[s], 0, None)) * DT_H)
        plan = greedy_plan(pr, dmd) if dmd > 0 else np.zeros(STEPS_PER_DAY)
        out.append(realised_cost(plan, pr, true_kw[s]))
    return float(np.mean(out)), out


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    source_pool, source_holdout = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)

    tdf = pd.read_csv("Data/uk_ev_features_full.csv")
    tXs, tXn, ty, tts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(tXs, tXn, ty, tts, holdout_days=HOLD)

    prices = price_days()
    Xs, Xn, y_ho, _ = source_holdout
    true_kw = inverse_target(y_ho, scalers)
    print(f"[e14] retention holdout: {len(true_kw)} steps = {len(true_kw)//96} days | "
          f"prices {prices.shape}", flush=True)

    def predict(model):
        p = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
        return np.clip(inverse_target(p, scalers), 0, None)

    rows = []

    # ---- anchors ----
    c_oracle, _ = cost_of(true_kw, true_kw, prices)
    last = np.concatenate([[true_kw[0]], true_kw[:-1]])      # naive persistence
    c_pers, _ = cost_of(last, true_kw, prices)
    keras.backend.clear_session()
    m0 = build_ev_model(); load_production_weights(m0, PROD)
    c_ref, _ = cost_of(predict(m0), true_kw, prices)
    for nm, c in (("oracle_demand", c_oracle), ("persistence", c_pers),
                  ("reference_no_adaptation", c_ref)):
        rows.append(dict(condition=nm, seed=-1, eur_per_day=round(c, 4)))
        print(f"[e14] {nm:24s} {c:8.3f} EUR/day", flush=True)

    # ---- adapted models ----
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
                                  source_train=source_pool,
                                  source_holdout=source_holdout)
            cost, _ = cost_of(predict(r["model"]), true_kw, prices)
            rows.append(dict(condition=name, seed=seed, eur_per_day=round(cost, 4),
                             retention_nrmse=float(r["source_retention_nrmse"])))
            print(f"[e14 {name} s={seed}] {cost:8.3f} EUR/day  "
                  f"(retention nRMSE {r['source_retention_nrmse']})", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== cost of forgetting, in euros ===")
    g = df[df.seed >= 0].groupby("condition").agg(
        eur=("eur_per_day", "mean"), sd=("eur_per_day", "std"),
        nrmse=("retention_nrmse", "mean"))
    g["vs_oracle_eur"] = g.eur - c_oracle
    print(g.round(4).to_string())
    print(f"\n  oracle {c_oracle:.3f} | persistence {c_pers:.3f} | "
          f"un-adapted {c_ref:.3f} EUR/day")
    if {"B3_warm", "B5_replay"} <= set(g.index):
        de = g.loc["B3_warm", "eur"] - g.loc["B5_replay", "eur"]
        dn = g.loc["B3_warm", "nrmse"] - g.loc["B5_replay", "nrmse"]
        print(f"  B3 - replay: {dn:+.4f} nRMSE  ->  {de:+.4f} EUR/day")
    print(f"\n[e14] wrote {OUT}")


if __name__ == "__main__":
    main()
