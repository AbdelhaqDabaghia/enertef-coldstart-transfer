"""
e17_mpc_value.py -- what does the Service 1 MPC actually save per day?

Every earlier experiment measured how FORECAST QUALITY moves the controller's
objective. None measured the controller itself. This one answers: against an
unmanaged site, what does running solve_ecc_mpc buy?

BASELINE. u = 0: EV charging happens as it arrives, no shifting. This is the
site without the controller.

TREATMENTS.
  mpc_perfect     u* solved on the REALISED demand -- the controller's ceiling
  mpc_production  u* solved on the deployed model's forecast -- what is running

Both are settled against the demand and PV that actually occurred, so the
comparison is honest: the plan is decided on a forecast, the bill is paid on
reality.

TWO LEDGERS, because they do not agree.
  objective   W_IMPORT*import + W_EXPORT*export      <- what the MPC minimises
              (self-consumption: export penalised twice as much as import)
  euros       price_spot * import - EXPORT_TARIFF * export
              (what the site pays; EXPORT_TARIFF configurable, 0 = surplus lost)

The controller optimises the first. Whether that also lowers the second is an
open question and precisely what this measures -- a controller tuned for
self-consumption can raise the bill when prices are volatile.

Writes Data/results/e17_mpc_value.csv. No training, no GPU.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import cvxpy as cp

from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model, load_production_weights

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SITE = "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv"
PRICE_CSV = os.environ.get("PRICE_CSV", "/mnt/c/dev/MPC_RL/BESS_MPC_PROJECT/prices_da_real.csv")
OUT = os.environ.get("OUT", "Data/results/e17_mpc_value.csv")

RET_DAYS = int(os.environ.get("RET_DAYS", "21"))   # longer window: more days
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
EXPORT_TARIFF = float(os.environ.get("EXPORT_TARIFF", "0.0"))  # EUR/kWh sold back

DT_H, H = 0.25, 96
U_MAX, EV_MAX, RAMP_MAX = 200.0, 800.0, 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 1.0, 2.0, 0.01, 0.2


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


def settle(u, ev_real, pv_real, price):
    """Both ledgers for one day, given the planned deviation u."""
    pg = (ev_real + u) - pv_real
    imp = np.clip(pg, 0, None)
    exp = np.clip(-pg, 0, None)
    imp_kwh, exp_kwh = float(imp.sum() * DT_H), float(exp.sum() * DT_H)
    objective = W_IMPORT * imp_kwh + W_EXPORT * exp_kwh
    euros = float(np.sum(price * imp * DT_H) - EXPORT_TARIFF * np.sum(price * 0 + exp * DT_H) * 0)
    euros = float(np.sum(price * imp * DT_H)) - EXPORT_TARIFF * exp_kwh
    return objective, imp_kwh, exp_kwh, euros


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sdf = pd.read_csv("Data/lux_source_features.csv")
    tail = (POOL_DAYS + RET_DAYS) * 96 + LOOKBACK
    a_, b_, c_, d_ = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
    _, ho = split_holdout(a_, b_, c_, d_, holdout_days=RET_DAYS)
    Xs, Xn, y_ho, ts_ho = ho
    ev_real = inverse_target(y_ho, scalers)

    site = pd.read_csv(SITE, parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    pv_real = (site.set_index("t")["PV_TotalProduction_kW"].astype(float)
               .reindex(pd.to_datetime(ts_ho, utc=True)).ffill().fillna(0.0).to_numpy())

    p = pd.read_csv(PRICE_CSV)
    col = "price_eur_kwh" if "price_eur_kwh" in p.columns else "price"
    pv_ = p[col].astype(float).to_numpy()
    if pv_.mean() > 5:
        pv_ = pv_ / 1000.0
    prices = pv_[:(len(pv_) // H) * H].reshape(-1, H)

    n = (len(ev_real) // H) * H
    ev_real, pv_real = ev_real[:n], pv_real[:n]
    ndays = n // H

    m = build_ev_model(); load_production_weights(m, PROD)
    pred = np.clip(inverse_target(
        np.clip(m.predict([Xs, Xn], verbose=0).reshape(-1), 0, None), scalers), 0, None)[:n]

    print(f"[e17] {ndays} days | EV mean {ev_real.mean():.1f} kW | "
          f"PV mean {pv_real.mean():.1f} kW | export tariff {EXPORT_TARIFF} EUR/kWh",
          flush=True)

    rows = []
    for d in range(ndays):
        s = slice(d * H, (d + 1) * H)
        ev, pv, pr = ev_real[s], pv_real[s], prices[d % len(prices)]
        for tag, u in (("unmanaged", np.zeros(H)),
                       ("mpc_perfect", solve_u(pv, ev)),
                       ("mpc_production", solve_u(pv, pred[s]))):
            o, i, e, eur = settle(u, ev, pv, pr)
            rows.append(dict(day=d, strategy=tag, objective=round(o, 3),
                             import_kwh=round(i, 3), export_kwh=round(e, 3),
                             euros=round(eur, 4)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    g = df.groupby("strategy")[["objective", "import_kwh", "export_kwh", "euros"]].mean()
    g = g.reindex(["unmanaged", "mpc_production", "mpc_perfect"])
    print("\n=== per day, averaged over", ndays, "days ===")
    print(g.round(3).to_string())
    base = g.loc["unmanaged"]
    print("\n=== what the MPC saves per day, vs the unmanaged site ===")
    for k in ["mpc_production", "mpc_perfect"]:
        r = g.loc[k]
        print(f"  {k:16s} objective {base.objective - r.objective:+8.2f}  "
              f"import {base.import_kwh - r.import_kwh:+8.2f} kWh  "
              f"export {base.export_kwh - r.export_kwh:+8.2f} kWh  "
              f"EUROS {base.euros - r.euros:+8.3f}")
    print("\n  positive = saving. Note the two ledgers can disagree in sign:")
    print("  the controller minimises grid exchange, not the bill.")
    print(f"\n[e17] wrote {OUT}")


if __name__ == "__main__":
    main()
