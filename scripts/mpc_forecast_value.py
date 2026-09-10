"""
mpc_forecast_value.py -- does a better forecast actually lower the energy bill?

The paper claims forecasting improvements propagate to EV charging decisions. That
claim is only supported if the DECISION is evaluated, not the forecast. This
harness closes the loop on the real site data.

METHOD (day-ahead, the regime the MPC is deployed in):
  for each test day
      build a 24 h PV forecast with each forecaster
      solve the site MPC on that forecast    -> charging plan u(t)
      settle the plan against the REAL PV    -> realised cost
  compare realised costs

CRITICAL: the MPC's own `cost` field is the QP objective, computed against the
FORECAST. Comparing those numbers across forecasters is meaningless -- an
optimistic forecast would "win" by planning against PV that never arrives. We
therefore recompute the cost from the plan and the realised PV:

    grid_import(t) = max(0, sum_n u_n(t) - PV_real(t))
    cost           = sum_t price(t) * grid_import(t) * dt
    (surplus export credited at EXPORT_RATIO * price, configurable)

FORECASTERS (all produce a 96-step ahead PV profile):
  perfect      : the realised PV -- the lower bound on achievable cost
  persistence  : yesterday's same-time profile (the baseline that beats our
                 models at the 15-min horizon, see persistence_baseline.csv)
  flat         : last observed value held constant
  model        : the trained forecaster  [TODO: wire the deployed PV model]

Reports the realised cost of each, and the gap each closes towards `perfect`.
That gap -- not nRMSE -- is the operationally meaningful quantity.

Run:  python -m scripts.mpc_forecast_value
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

MPC_DIR = os.environ.get("MPC_DIR", r"/mnt/c/dev/Enertef_MPC_Controller")
SITE_CSV = os.environ.get("SITE_CSV", "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv")
OUT = os.environ.get("OUT", "Data/results/mpc_forecast_value.csv")

STEPS_PER_DAY = 96
DT_H = 0.25
TEST_DAYS = int(os.environ.get("TEST_DAYS", "14"))
PRICE_EUR_KWH = float(os.environ.get("PRICE", "0.15"))   # TODO: real ENTSO-E series
EXPORT_RATIO = float(os.environ.get("EXPORT_RATIO", "0.0"))
N_VEHICLES = int(os.environ.get("N_VEHICLES", "22"))


# ---------------------------------------------------------------- forecasters
def fc_perfect(pv_real_day, hist):
    return pv_real_day.copy()


def fc_persistence(pv_real_day, hist):
    """Same time yesterday -- the seasonal-naive baseline."""
    return hist[-STEPS_PER_DAY:].copy()


def fc_flat(pv_real_day, hist):
    return np.full(STEPS_PER_DAY, hist[-1], dtype=float)


FORECASTERS = {"perfect": fc_perfect, "persistence": fc_persistence, "flat": fc_flat}


# ------------------------------------------------------- demand forecasters
# The EV demand forecast decides HOW MUCH to schedule. Its error is settled by
# the rule in realised_cost_with_demand(): the real demand must be served, so
# whatever was not planned is charged as it arrives, at the prevailing price.
# Under-forecasting therefore loses the cheap hours; over-forecasting is only
# lightly penalised, since surplus plan is simply not executed. That asymmetry
# is physical, not an artefact.
def dfc_perfect(ev_real_day, ev_hist):
    return float(ev_real_day.sum() * DT_H)


def dfc_persistence(ev_real_day, ev_hist):
    return float(ev_hist[-STEPS_PER_DAY:].sum() * DT_H)


def dfc_weekly(ev_real_day, ev_hist):
    """Same weekday last week -- the stronger seasonal-naive baseline for EV."""
    if len(ev_hist) < 7 * STEPS_PER_DAY:
        return dfc_persistence(ev_real_day, ev_hist)
    return float(ev_hist[-7 * STEPS_PER_DAY:-6 * STEPS_PER_DAY].sum() * DT_H)


DEMAND_FORECASTERS = {"perfect": dfc_perfect, "persistence": dfc_persistence,
                      "weekly": dfc_weekly}


def realised_cost_with_demand(plan, pv_real, price, ev_real):
    """Settle a plan sized on a FORECAST demand against the demand that arrived.

    plan     : (T,) planned site charging power [kW]
    ev_real  : (T,) realised EV demand [kW] -- must be served in full
    Energy planned up to the real total is delivered at the planned times; any
    shortfall is served on the realised demand profile (i.e. as it arrives).
    """
    e_plan = float(np.sum(plan) * DT_H)
    e_real = float(np.sum(ev_real) * DT_H)
    if e_plan <= 0:
        served = np.zeros(STEPS_PER_DAY)
    else:
        served = plan * min(1.0, e_real / e_plan)      # cannot charge absent cars
    shortfall = max(0.0, e_real - float(np.sum(served) * DT_H))
    if shortfall > 0 and ev_real.sum() > 0:
        served = served + ev_real * (shortfall / (float(np.sum(ev_real) * DT_H)))
    net = served - np.asarray(pv_real, float)
    imp, exp = np.clip(net, 0, None), np.clip(-net, 0, None)
    return float(np.sum(price * imp * DT_H) - EXPORT_RATIO * np.sum(price * exp * DT_H))


# ---------------------------------------------------------------- settlement
def realised_cost(u_plan, pv_real, price):
    """Settle a charging plan against the PV that actually arrived.

    u_plan : (T,) total site charging power [kW] decided from the forecast
    pv_real: (T,) realised PV [kW]
    price  : (T,) EUR/kWh
    """
    net = np.asarray(u_plan, float) - np.asarray(pv_real, float)
    imp = np.clip(net, 0, None)
    exp = np.clip(-net, 0, None)
    return float(np.sum(price * imp * DT_H) - EXPORT_RATIO * np.sum(price * exp * DT_H))


# ---------------------------------------------------------------- MPC wrapper
def load_mpc():
    """Import the deployed SiteMPC. Kept isolated so the harness degrades to a
    greedy reference if the controller cannot be imported here."""
    sys.path.insert(0, os.path.dirname(MPC_DIR))
    pkg = os.path.basename(MPC_DIR)
    mod = __import__(f"{pkg}.mpc_site", fromlist=["SiteMPC"])
    cfgm = __import__(f"{pkg}.config", fromlist=["MPCConfig", "TimeConfig",
                                                 "DegradationConfig", "VehicleSpec"])
    return mod.SiteMPC, cfgm


def greedy_plan(pv_forecast, price, demand_kwh):
    """Fallback scheduler: place the day's charging demand in the cheapest net
    hours given the forecast (price minus forecast PV value). Not the deployed
    MPC, but a transparent stand-in that still USES the forecast, so the
    forecast-value comparison remains meaningful."""
    score = price - 1e-3 * np.asarray(pv_forecast, float)
    order = np.argsort(score)
    plan = np.zeros(STEPS_PER_DAY)
    cap = float(os.environ.get("CHARGER_CAP_KW", "200"))
    remaining = demand_kwh
    for i in order:
        if remaining <= 0:
            break
        take = min(cap * DT_H, remaining)
        plan[i] = take / DT_H
        remaining -= take
    return plan


# ---------------------------------------------------------------- main
def main():
    df = pd.read_csv(SITE_CSV, parse_dates=["Started at"])
    pv = df["PV_TotalProduction_kW"].astype(float).ffill().fillna(0.0).to_numpy()
    ev = df["EV_TotalConsumption_kW"].astype(float).ffill().fillna(0.0).to_numpy()

    n_days = len(pv) // STEPS_PER_DAY
    test_start = n_days - TEST_DAYS
    print(f"[mpc] {len(pv)} steps, {n_days} days | testing the last {TEST_DAYS}")

    # Price profile. Flat by default; PRICE_CSV supplies a real day-ahead series.
    # NOTE: if the series does not overlap the site window, it is used as a
    # REPRESENTATIVE profile (day d of the series applied to test day d) -- valid
    # for asking whether realistic price variability widens the forecast-value
    # margin, NOT for quoting an absolute euro figure. Declare this in the paper.
    price_days = None
    pcsv = os.environ.get("PRICE_CSV", "")
    if pcsv:
        pdf = pd.read_csv(pcsv)
        col = "price_eur_kwh" if "price_eur_kwh" in pdf.columns else "price"
        v = pdf[col].astype(float).to_numpy()
        if v.mean() > 5:                     # looks like EUR/MWh
            v = v / 1000.0
        nd = len(v) // STEPS_PER_DAY
        price_days = v[:nd * STEPS_PER_DAY].reshape(nd, STEPS_PER_DAY)
        print(f"[mpc] prices from {pcsv}: {nd} days, "
              f"mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f} EUR/kWh")
    price = np.full(STEPS_PER_DAY, PRICE_EUR_KWH)
    rows = []
    for d in range(test_start, n_days):
        s, e = d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY
        pv_day, ev_day = pv[s:e], ev[s:e]
        hist = pv[:s]
        if len(hist) < STEPS_PER_DAY:
            continue
        demand_kwh = float(ev_day.sum() * DT_H)     # the day's real charging need
        if demand_kwh <= 0:
            continue
        if price_days is not None:
            price = price_days[(d - test_start) % len(price_days)]

        ev_hist = ev[:s]
        # Cross PV forecast x demand forecast, so the value of each can be
        # attributed separately (perfect/perfect is the achievable lower bound).
        for pv_name, pv_fn in FORECASTERS.items():
            fc = np.clip(pv_fn(pv_day, hist), 0, None)
            for dm_name, dm_fn in DEMAND_FORECASTERS.items():
                dmd = max(0.0, dm_fn(ev_day, ev_hist))
                plan = greedy_plan(fc, price, dmd) if dmd > 0 else np.zeros(STEPS_PER_DAY)
                cost = realised_cost_with_demand(plan, pv_day, price, ev_day)
                rows.append(dict(day=d, pv_forecaster=pv_name,
                                 demand_forecaster=dm_name,
                                 pv_nrmse=round(float(
                                     np.sqrt(np.mean((fc - pv_day) ** 2)) /
                                     (pv_day.mean() + 1e-9)), 4),
                                 demand_pred_kwh=round(dmd, 2),
                                 demand_real_kwh=round(demand_kwh, 2),
                                 realised_cost_eur=round(cost, 3)))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    piv = out.pivot_table(index="pv_forecaster", columns="demand_forecaster",
                          values="realised_cost_eur", aggfunc="mean").round(3)
    print("\nRealised cost (EUR/day)  [rows: PV forecast, cols: demand forecast]")
    print(piv.to_string())

    base = piv.loc["perfect", "perfect"]
    print(f"\nPerfect foresight on both: {base:.3f} EUR/day")
    print("Cost of forecast error (EUR/day above that bound):")
    print(f"  PV only wrong (persistence PV, perfect demand)  "
          f"{piv.loc['persistence','perfect'] - base:+8.3f}")
    print(f"  demand only wrong (perfect PV, persistence dmd) "
          f"{piv.loc['perfect','persistence'] - base:+8.3f}")
    print(f"  both wrong (persistence, persistence)           "
          f"{piv.loc['persistence','persistence'] - base:+8.3f}")
    print(f"  demand weekly-naive (perfect PV)                "
          f"{piv.loc['perfect','weekly'] - base:+8.3f}")
    print(f"\n[mpc] wrote {OUT}")
    print("[mpc] NOTE: greedy stand-in scheduler and flat price. Wire SiteMPC and "
          "the ENTSO-E series before quoting any number in the paper.")


if __name__ == "__main__":
    main()
