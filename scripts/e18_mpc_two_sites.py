"""
e18_mpc_two_sites.py -- the REAL Service 1 controller, planned vs realised.

e17 implemented the wrong objective. The savings figure reported in
ECC_MCP_v0.5.ipynb comes from `solve_ecc_mpc_two_sites`, not `solve_ecc_mpc`:

    two_sites   price-driven, 8*cost_import - 2*revenue_export, prices in
                EUR/MWh, TWO metering points (EMOB1 u1, EMOB2 u2) each with
                |u| <= 200 kW -> 400 kW of flexibility, sum(u1)=sum(u2)=0
    solve_ecc   self-consumption, import + 2*export, no prices, one site,
                |u| <= 200 kW

This script transcribes the two-site function verbatim and then asks three
questions the notebook does not separate.

  A. PLANNED savings, synthetic prices  -- reproduces the notebook exactly.
     Both legs of the notebook's subtraction are forecasts: cost_base uses
     ev_base_fc and cost_opt uses the solver's planned import. Nothing is
     settled against what happened.

  B. PLANNED savings, REAL day-ahead prices. The notebook's price_fc is
     synthetic_price_curve(base=80, peak=220, night=30): a fabricated 30->220
     EUR/MWh spread. Real LU day-ahead over the same season is far narrower,
     and arbitrage value scales with the spread.

  C. REALISED savings, real prices. The plan u* is decided on the forecast and
     then settled against the demand and PV that actually arrived, against an
     unmanaged site settled the same way. This is what the site banks.

A - C is the overstatement. The gap is the finding.

Writes Data/results/e18_mpc_two_sites.csv. No training, no GPU.
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
OUT = os.environ.get("OUT", "Data/results/e18_mpc_two_sites.csv")

RET_DAYS = int(os.environ.get("RET_DAYS", "21"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))

DT_H, H = 0.25, 96

# --- transcribed from solve_ecc_mpc_two_sites (ECC_MCP_v0.5.ipynb, cell 48) ---
U_LO, U_HI = -200.0, 200.0
EV1_LO, EV1_HI = 0.0, 1200.0
EV2_LO, EV2_HI = 0.0, 1800.0
RAMP_MAX = 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 8.0, 2.0, 1e-4, 0.01

# capacity split, notebook cell 48
CAP1 = 4 * 300.0
CAP2 = 4 * 400.0 + 8 * 22.0
SHARE1 = CAP1 / (CAP1 + CAP2)
SHARE2 = 1.0 - SHARE1


def synthetic_price_curve(hours, base=80.0, peak=220.0, night=30.0):
    """Verbatim from the notebook (noise_std=0 as called there)."""
    p = np.full(len(hours), base, dtype=float)
    p[(hours >= 0) & (hours < 6)] = night
    p[((hours >= 7) & (hours < 10)) | ((hours >= 17) & (hours < 20))] = peak
    return p


def solve_two_sites(pv, ev1b, ev2b, price):
    """The deployed objective. price in EUR/MWh. Returns (u1, u2)."""
    n = len(pv)
    u1, u2 = cp.Variable(n), cp.Variable(n)
    ev1c, ev2c = ev1b + u1, ev2b + u2
    pgrid = ev1c + ev2c - pv
    imp, exp = cp.Variable(n), cp.Variable(n)

    cons = [imp >= 0, exp >= 0, pgrid == imp - exp,
            ev1c >= EV1_LO, ev1c <= EV1_HI, ev2c >= EV2_LO, ev2c <= EV2_HI,
            u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]

    e_imp = imp * DT_H / 1000.0
    e_exp = exp * DT_H / 1000.0
    obj = (W_IMPORT * cp.sum(cp.multiply(price, e_imp))
           - W_EXPORT * cp.sum(cp.multiply(price, e_exp))
           + W_U * (cp.sum_squares(u1) + cp.sum_squares(u2))
           + W_RAMP * (cp.sum_squares(u1[1:] - u1[:-1])
                       + cp.sum_squares(u2[1:] - u2[:-1])))
    try:
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return np.zeros(n), np.zeros(n)
    if u1.value is None or u2.value is None:
        return np.zeros(n), np.zeros(n)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


def bill(ev_total, pv, price):
    """The notebook's ledger: import cost minus export revenue, both at 1:1."""
    pg = ev_total - pv
    imp = np.clip(pg, 0, None)
    exp = np.clip(-pg, 0, None)
    return float(np.sum(imp * DT_H / 1000.0 * price)
                 - np.sum(exp * DT_H / 1000.0 * price))


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
    ts = pd.to_datetime(ts_ho, utc=True)
    pv_real = (site.set_index("t")["PV_TotalProduction_kW"].astype(float)
               .reindex(ts).ffill().fillna(0.0).to_numpy())

    p = pd.read_csv(PRICE_CSV)
    col = "price" if "price" in p.columns else "price_eur_kwh"
    v = p[col].astype(float).to_numpy()
    if v.mean() < 5:                      # stored in EUR/kWh -> EUR/MWh
        v = v * 1000.0
    real_prices = v[:(len(v) // H) * H].reshape(-1, H)

    n = (len(ev_real) // H) * H
    ev_real, pv_real, ts = ev_real[:n], pv_real[:n], ts[:n]
    ndays = n // H
    hours = ts.hour.to_numpy()

    m = build_ev_model(); load_production_weights(m, PROD)
    ev_fc = np.clip(inverse_target(
        np.clip(m.predict([Xs, Xn], verbose=0).reshape(-1), 0, None), scalers), 0, None)[:n]

    print(f"[e18] {ndays} days | EV mean real {ev_real.mean():.1f} kW / "
          f"forecast {ev_fc.mean():.1f} kW | PV mean {pv_real.mean():.1f} kW", flush=True)
    print(f"[e18] real price EUR/MWh: mean {v.mean():.1f} min {v.min():.1f} "
          f"max {v.max():.1f} | synthetic: 30 / 80 / 220", flush=True)

    rows = []
    for d in range(ndays):
        s = slice(d * H, (d + 1) * H)
        evR, pvR = ev_real[s], pv_real[s]
        evF = ev_fc[s]
        f1, f2 = SHARE1 * evF, SHARE2 * evF
        r1, r2 = SHARE1 * evR, SHARE2 * evR

        for tag, price in (("synthetic", synthetic_price_curve(hours[s])),
                           ("real", real_prices[d % len(real_prices)])):
            u1, u2 = solve_two_sites(pvR, f1, f2, price)

            # A/B -- the notebook's comparison: both legs on the forecast
            planned_base = bill(evF, pvR, price)
            planned_opt = bill(np.clip(f1 + u1, 0, None) + np.clip(f2 + u2, 0, None),
                               pvR, price)

            # C -- settle the same plan against what actually arrived
            real_base = bill(evR, pvR, price)
            real_opt = bill(np.clip(r1 + u1, 0, None) + np.clip(r2 + u2, 0, None),
                            pvR, price)

            rows.append(dict(day=d, prices=tag,
                             planned_base=round(planned_base, 4),
                             planned_opt=round(planned_opt, 4),
                             planned_savings=round(planned_base - planned_opt, 4),
                             realised_base=round(real_base, 4),
                             realised_opt=round(real_opt, 4),
                             realised_savings=round(real_base - real_opt, 4)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    g = df.groupby("prices")[["planned_base", "planned_opt", "planned_savings",
                              "realised_base", "realised_opt", "realised_savings"]].mean()
    print("\n=== EUR per day, averaged over", ndays, "days ===")
    print(g.round(3).to_string())
    print("\n=== the three questions ===")
    for tag in ("synthetic", "real"):
        if tag not in g.index:
            continue
        r = g.loc[tag]
        print(f"  {tag:10s} prices: PLANNED {r.planned_savings:+8.2f} EUR/day   "
              f"REALISED {r.realised_savings:+8.2f} EUR/day   "
              f"overstatement {r.planned_savings - r.realised_savings:+8.2f}")
    print("\n  PLANNED is the notebook's number: forecast minus forecast.")
    print("  REALISED settles the same plan against the demand that arrived.")
    print(f"\n[e18] wrote {OUT}")


if __name__ == "__main__":
    main()
