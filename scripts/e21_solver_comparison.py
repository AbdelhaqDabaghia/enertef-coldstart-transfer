"""
e21_solver_comparison.py -- does the deployed MPC actually optimise?

THE OBSERVATION (from the dashboard session). historical.kpi_validation shows
cost_baseline == cost_optimised to within 1.59e-14 % across ~12 consecutive
cycles, so savings_eur = 0.00 EUR every time. An exact zero is not a weak
optimisation; it is the signature of u = 0.

THE SUSPICION. realtime_runner.py does NOT use the convex solver the notebook
uses. ECC_MCP_v0.5.ipynb solves this as a convex QP with cvxpy/OSQP.
realtime_runner re-implements the same problem -- identical weights
(w_import=8, w_export=2, w_u=1e-4, w_ramp=0.01), identical bounds (+/-200 kW,
ramp 80) -- with scipy SLSQP, a general nonlinear solver, on 2n = 192 variables
with np.abs(np.diff(x)) constraints that are not differentiable, starting from

    x0 = np.zeros(2 * n)

and it uses the result whether or not the solve converged:

    if not result_obj.success:
        print('[MPC] Solver did not fully converge: ...')
    x_opt = result_obj.x          # used anyway

If SLSQP makes no progress it returns x0, which is exactly zero, which produces
exactly zero savings and a warning in a log nobody reads.

THE TEST. Same day, same prices, same forecast, same weights. Solve it both
ways and compare. If OSQP finds a saving where SLSQP returns zero, the
deployed controller is not optimising and the problem is the solver, not the
site, the tariff, or the forecast.

Writes Data/results/corrected_causal_pipeline/e21_solver_comparison.csv
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from coldstart_transfer.features import engineer_features, WEATHER_VARS
from coldstart_transfer.windowing import (make_windows, split_holdout,
                                          inverse_target, LOOKBACK)

LUX = "Data/lux_source_features.csv"
SITE = "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv"
PRICE_CSV = os.environ.get(
    "PRICE_CSV", "/mnt/c/dev/MPC_RL/BESS_MPC_PROJECT/prices_da_real.csv")
OUT = "Data/results/corrected_causal_pipeline/e21_solver_comparison.csv"
RET_DAYS = 7

DT_H, H = 0.25, 96
U_MIN, U_MAX = -200.0, 200.0
EV1_MIN, EV1_MAX = 0.0, 1200.0
EV2_MIN, EV2_MAX = 0.0, 1800.0
RAMP_MAX = 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 8.0, 2.0, 1e-4, 0.01
CAP1, CAP2 = 4 * 300.0, 4 * 400.0 + 8 * 22.0
SHARE1 = CAP1 / (CAP1 + CAP2)
SHARE2 = 1.0 - SHARE1


def objective_value(u1, u2, ev1b, ev2b, pv, price):
    """The deployed objective, evaluated identically for both solvers."""
    pgrid = (ev1b + u1) + (ev2b + u2) - pv
    imp = np.maximum(pgrid, 0) * DT_H / 1000.0
    exp = np.maximum(-pgrid, 0) * DT_H / 1000.0
    return float(W_IMPORT * np.sum(price * imp)
                 - W_EXPORT * np.sum(price * exp)
                 + W_U * (np.sum(u1 ** 2) + np.sum(u2 ** 2))
                 + W_RAMP * (np.sum(np.diff(u1) ** 2) + np.sum(np.diff(u2) ** 2)))


def solve_slsqp(pv, ev1b, ev2b, price):
    """Transcribed from realtime_runner.py, including x0 and the fallback."""
    n = len(pv)

    def split(x):
        return x[:n], x[n:]

    def obj(x):
        u1, u2 = split(x)
        return objective_value(u1, u2, ev1b, ev2b, pv, price)

    x0 = np.zeros(2 * n, dtype=np.float64)
    bounds = [(U_MIN, U_MAX)] * (2 * n)
    cons = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n])},
        {"type": "eq", "fun": lambda x: np.sum(x[n:])},
        {"type": "ineq", "fun": lambda x: (ev1b + x[:n]) - EV1_MIN},
        {"type": "ineq", "fun": lambda x: EV1_MAX - (ev1b + x[:n])},
        {"type": "ineq", "fun": lambda x: (ev2b + x[n:]) - EV2_MIN},
        {"type": "ineq", "fun": lambda x: EV2_MAX - (ev2b + x[n:])},
        {"type": "ineq", "fun": lambda x: RAMP_MAX - np.abs(np.diff(x[:n]))},
        {"type": "ineq", "fun": lambda x: RAMP_MAX - np.abs(np.diff(x[n:]))},
    ]
    r = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                 options={"maxiter": 200, "ftol": 1e-3})
    u1, u2 = split(r.x)
    return u1, u2, bool(r.success), str(r.message), int(r.nit)


def solve_osqp(pv, ev1b, ev2b, price):
    """The notebook's convex formulation."""
    n = len(pv)
    u1, u2 = cp.Variable(n), cp.Variable(n)
    ev1c, ev2c = ev1b + u1, ev2b + u2
    pgrid = ev1c + ev2c - pv
    imp, exp = cp.Variable(n), cp.Variable(n)
    cons = [imp >= 0, exp >= 0, pgrid == imp - exp,
            ev1c >= EV1_MIN, ev1c <= EV1_MAX,
            ev2c >= EV2_MIN, ev2c <= EV2_MAX,
            u1 >= U_MIN, u1 <= U_MAX, u2 >= U_MIN, u2 <= U_MAX,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]
    e_i, e_e = imp * DT_H / 1000.0, exp * DT_H / 1000.0
    obj = (W_IMPORT * cp.sum(cp.multiply(price, e_i))
           - W_EXPORT * cp.sum(cp.multiply(price, e_e))
           + W_U * (cp.sum_squares(u1) + cp.sum_squares(u2))
           + W_RAMP * (cp.sum_squares(u1[1:] - u1[:-1])
                       + cp.sum_squares(u2[1:] - u2[:-1])))
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=cp.OSQP, verbose=False)
    if u1.value is None:
        return np.zeros(n), np.zeros(n), False, str(p.status), -1
    return (np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel(),
            p.status in ("optimal", "optimal_inaccurate"), str(p.status), -1)


def bill(ev_total, pv, price):
    pg = ev_total - pv
    return float(np.sum(np.maximum(pg, 0) * DT_H / 1000.0 * price)
                 - np.sum(np.maximum(-pg, 0) * DT_H / 1000.0 * price))


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    raw = pd.read_csv(LUX)[["timestamp", "ev_kw"] + WEATHER_VARS]
    feat = engineer_features(raw, mode="causal")
    Xs, Xn, y, ts = make_windows(feat, scalers)
    _, ho = split_holdout(Xs, Xn, y, ts, holdout_days=RET_DAYS)
    _, _, hy, hts = ho
    ev = inverse_target(hy, scalers)
    n = (len(ev) // H) * H
    ev, hts = ev[:n], pd.to_datetime(hts[:n], utc=True)

    site = pd.read_csv(SITE, parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    pv_all = (site.set_index("t")["PV_TotalProduction_kW"].astype(float)
              .reindex(hts).ffill().fillna(0.0).to_numpy())

    p = pd.read_csv(PRICE_CSV)
    col = "price" if "price" in p.columns else "price_eur_kwh"
    v = p[col].astype(float).to_numpy()
    if v.mean() < 5:
        v = v * 1000.0
    prices = v[:(len(v) // H) * H].reshape(-1, H)

    rows = []
    for d in range(n // H):
        s = slice(d * H, (d + 1) * H)
        evd, pvd, pr = ev[s], pv_all[s], prices[d % len(prices)]
        e1, e2 = SHARE1 * evd, SHARE2 * evd
        zero = np.zeros(H)

        j0 = objective_value(zero, zero, e1, e2, pvd, pr)
        b0 = bill(evd, pvd, pr)

        s1, s2, sok, smsg, snit = solve_slsqp(pvd, e1, e2, pr)
        js = objective_value(s1, s2, e1, e2, pvd, pr)
        bs = bill(np.clip(e1 + s1, 0, None) + np.clip(e2 + s2, 0, None), pvd, pr)

        o1, o2, ook, omsg, _ = solve_osqp(pvd, e1, e2, pr)
        jo = objective_value(o1, o2, e1, e2, pvd, pr)
        bo = bill(np.clip(e1 + o1, 0, None) + np.clip(e2 + o2, 0, None), pvd, pr)

        rows.append(dict(
            day=d, baseline_eur=round(b0, 3), baseline_obj=round(j0, 3),
            slsqp_converged=sok, slsqp_iters=snit,
            slsqp_max_abs_u=round(float(np.max(np.abs(np.r_[s1, s2]))), 6),
            slsqp_obj=round(js, 3), slsqp_eur=round(bs, 3),
            slsqp_saving_eur=round(b0 - bs, 4),
            osqp_status=omsg,
            osqp_max_abs_u=round(float(np.max(np.abs(np.r_[o1, o2]))), 3),
            osqp_obj=round(jo, 3), osqp_eur=round(bo, 3),
            osqp_saving_eur=round(b0 - bo, 4),
            slsqp_msg=smsg[:60]))
        print("[e21] day %d  SLSQP conv=%-5s max|u|=%8.4f saving=%7.3f EUR | "
              "OSQP %-8s max|u|=%7.1f saving=%7.3f EUR"
              % (d, sok, rows[-1]["slsqp_max_abs_u"], rows[-1]["slsqp_saving_eur"],
                 omsg, rows[-1]["osqp_max_abs_u"], rows[-1]["osqp_saving_eur"]),
              flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== objective achieved (lower is better) ===")
    print("  do nothing (u=0) : %10.2f" % df.baseline_obj.mean())
    print("  SLSQP (deployed) : %10.2f" % df.slsqp_obj.mean())
    print("  OSQP  (notebook) : %10.2f" % df.osqp_obj.mean())
    print("\n=== realised saving, EUR/day ===")
    print("  SLSQP (deployed) : %+8.3f" % df.slsqp_saving_eur.mean())
    print("  OSQP  (notebook) : %+8.3f" % df.osqp_saving_eur.mean())
    print("\n=== how far each solver moved the setpoints ===")
    print("  SLSQP max|u| : %.6f kW  (converged on %d of %d days)"
          % (df.slsqp_max_abs_u.mean(), int(df.slsqp_converged.sum()), len(df)))
    print("  OSQP  max|u| : %.1f kW" % df.osqp_max_abs_u.mean())
    if df.slsqp_max_abs_u.max() < 1e-3:
        print("\n  SLSQP NEVER MOVED FROM x0 = 0. The deployed controller is not "
              "optimising;\n  it returns its initial guess and reports zero "
              "savings, which is exactly\n  what historical.kpi_validation "
              "shows.")
    print("\n[e21] wrote %s" % OUT)


if __name__ == "__main__":
    main()
