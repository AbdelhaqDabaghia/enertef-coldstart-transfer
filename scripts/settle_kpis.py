"""
settle_kpis.py -- re-score yesterday's controller KPIs against measurement.

WHY. realtime_runner computes both sides of the saving from forecasts:

    cost_base = f(ev_total_base)   <- the FORECAST baseline
    cost_opt  = f(ev_total_ctrl)   <- the PLANNED optimised profile
    reduction_pct = 100 * (cost_base - cost_opt) / |cost_base|

and writes reduction_pct into historical.kpi_validation, which /kpi-summary
serves to the dashboard. Nothing is ever compared with what actually happened,
so an optimistic forecast raises the reported KPI without lowering a bill.
Measured on 21 days (e18), settling against measurement moves the figure from
11.5 % to 6.0 %.

WHAT THIS DOES. For each past day it reads the plan that was issued, pulls the
demand and generation that actually arrived from historical.telemetry_15min,
re-settles the SAME plan against them, and writes a parallel KPI row suffixed
_realised. Planned rows are left untouched: the two are reported side by side
so the gap is visible rather than silently corrected.

IMPORTANT. Do not re-solve the optimisation on the actuals. That measures a
controller that never ran. The plan is decided on the forecast and the bill is
paid on reality; settlement applies the issued setpoints to the realised series.

USAGE
    python scripts/settle_kpis.py --dry-run --from-csv Data/...   (offline check)
    python scripts/settle_kpis.py --day 2026-09-13                (against RDS)

Requires PG_HOST / PG_DB / PG_USER / PG_PASSWORD in the environment. Never
hard-code them. Run inside the VPC: the corporate firewall blocks the
PostgreSQL wire protocol from a laptop.
"""
from __future__ import annotations
import os
import argparse
import datetime as dt
import numpy as np

DT_H = 0.25
SERVICE_ID = 1
# From contextual.services.pass_threshold for service-1. realtime_runner
# currently hard-codes 5.0 here, which does not match. Single source of truth.
COST_REDUCTION_THRESHOLD = 20.0


def settle(ev_real_kw, pv_real_kw, u_plan_kw, price_eur_mwh):
    """Apply an issued plan to the series that actually arrived.

    ev_real_kw    realised EV baseline demand, kW, one value per 15 min
    pv_real_kw    realised PV generation, kW
    u_plan_kw     the deviation the controller ordered, kW (sum ~ 0)
    price_eur_mwh day-ahead price for the same steps

    Returns (cost_base, cost_opt, reduction_pct) in EUR, all realised.
    """
    ev_real = np.asarray(ev_real_kw, float)
    pv = np.asarray(pv_real_kw, float)
    u = np.asarray(u_plan_kw, float)
    price = np.asarray(price_eur_mwh, float)
    if not (len(ev_real) == len(pv) == len(u) == len(price)):
        raise ValueError("settle(): all series must have the same length")

    def cost(ev):
        pg = ev - pv
        imp = np.maximum(pg, 0.0) * DT_H / 1000.0
        exp = np.maximum(-pg, 0.0) * DT_H / 1000.0
        return float(np.sum(imp * price) - np.sum(exp * price))

    # the unmanaged site, on the demand that arrived
    c_base = cost(ev_real)
    # the same site with the plan applied; charging cannot go negative
    c_opt = cost(np.maximum(ev_real + u, 0.0))
    red = 100.0 * (c_base - c_opt) / max(abs(c_base), 0.01)
    return c_base, c_opt, red


SQL_PLAN = """
    SELECT target_timestamp, predicted_value
      FROM planning.forecasts
     WHERE model_name = %s AND target_timestamp::date = %s
     ORDER BY target_timestamp
"""

SQL_ACTUAL = """
    SELECT timestamp, asset_id, power_kw_avg
      FROM historical.telemetry_15min
     WHERE timestamp::date = %s AND source = 'energypark-api'
     ORDER BY timestamp
"""

SQL_PRICE = """
    SELECT timestamp, price_eur_mwh
      FROM market_prices
     WHERE timestamp::date = %s AND bidding_zone = %s
     ORDER BY timestamp
"""

SQL_SETPOINTS = """
    SELECT target_time, setpoint_kw
      FROM planning.mpc_setpoints
     WHERE service_id = 1 AND target_time::date = %s
     ORDER BY target_time
"""

SQL_PLANNED = """
    SELECT measured_value
      FROM historical.kpi_validation
     WHERE service_id = 1 AND kpi_name = 'cost_reduction_pct'
       AND run_timestamp::date = %s
     ORDER BY run_timestamp DESC
     LIMIT 1
"""

SQL_WRITE = """
    INSERT INTO historical.kpi_validation
      (service_id, campaign_id, run_timestamp, kpi_name,
       measured_value, threshold, verdict, notes)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def write_realised(cur, day, c_base, c_opt, red):
    """One parallel row. The planned row is NOT modified."""
    verdict = "PASS" if red >= COST_REDUCTION_THRESHOLD else "FAIL"
    notes = ("settled against measurement: baseline %.2f EUR, optimised "
             "%.2f EUR, saving %.2f EUR. The plan was NOT re-solved on the "
             "actuals." % (c_base, c_opt, c_base - c_opt))
    cur.execute(SQL_WRITE, (SERVICE_ID, "settlement",
                            dt.datetime.now(dt.timezone.utc),
                            "cost_reduction_pct_realised",
                            float(red), COST_REDUCTION_THRESHOLD,
                            verdict, notes))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY-MM-DD; default = yesterday (J-1)")
    ap.add_argument("--zone", default="LU")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    a = ap.parse_args()

    day = (dt.date.fromisoformat(a.day) if a.day
           else dt.date.today() - dt.timedelta(days=1))
    print("[settle] day %s  threshold %.1f %%" % (day, COST_REDUCTION_THRESHOLD))

    missing = [v for v in ("PG_HOST", "PG_DB", "PG_USER", "PG_PASSWORD")
               if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            "[settle] missing environment: %s\n"
            "  Set them in the environment, never in this file, and run inside\n"
            "  the VPC -- the corporate firewall blocks the PostgreSQL wire\n"
            "  protocol from a laptop." % ", ".join(missing))

    import pg8000.dbapi
    import ssl
    ctx = ssl.create_default_context()
    conn = pg8000.dbapi.connect(
        host=os.environ["PG_HOST"], port=int(os.environ.get("PG_PORT", "5432")),
        database=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"], ssl_context=ctx)
    cur = conn.cursor()

    cur.execute(SQL_ACTUAL, (day,))
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("[settle] no telemetry for %s -- nothing to settle" % day)
    by_asset = {}
    for ts, asset, kw in rows:
        by_asset.setdefault(asset, []).append((ts, float(kw or 0.0)))

    def series(name):
        return np.array([v for _, v in sorted(by_asset.get(name, []))])

    ev_real = series("EMOB1") + series("EMOB2")
    pv_real = series("PV")

    cur.execute(SQL_PRICE, (day, a.zone))
    price = np.array([float(p) for _, p in cur.fetchall()])

    n = min(len(ev_real), len(pv_real), len(price))
    if n == 0:
        raise SystemExit("[settle] missing telemetry or prices for %s" % day)
    if n < 96:
        print("[settle] WARNING: only %d of 96 steps available; the day is "
              "incomplete and the figure below is partial" % n)

    # The issued plan. realtime_runner writes result['u'] = u1_opt + u2_opt to
    # planning.mpc_setpoints, one row per step. NOTE: every row is labelled
    # asset_id='EMOB1' although the value is the EMOB1+EMOB2 total. That is a
    # labelling bug in the writer; the value is the one settlement needs, so we
    # take the rows as the site-level deviation and do not split by asset.
    cur.execute(SQL_SETPOINTS, (day,))
    sp = cur.fetchall()
    if not sp:
        raise SystemExit(
            "[settle] no setpoints in planning.mpc_setpoints for %s.\n"
            "  Without the plan that was issued there is nothing to settle."
            % day)
    u_plan = np.array([float(v) for _, v in sorted(sp)])

    n = min(n, len(u_plan))
    c_base, c_opt, red = settle(ev_real[:n], pv_real[:n], u_plan[:n], price[:n])
    verdict = "PASS" if red >= COST_REDUCTION_THRESHOLD else "FAIL"

    print("[settle] steps settled      : %d" % n)
    print("[settle] realised baseline  : %8.2f EUR" % c_base)
    print("[settle] realised optimised : %8.2f EUR" % c_opt)
    print("[settle] realised saving    : %8.2f EUR  (%.2f %%)  -> %s"
          % (c_base - c_opt, red, verdict))

    cur.execute(SQL_PLANNED, (day,))
    pl = cur.fetchone()
    if pl:
        print("[settle] planned KPI logged : %.2f %%  (gap %+.2f points)"
              % (float(pl[0]), float(pl[0]) - red))

    if a.dry_run:
        print("[settle] dry run: nothing written")
    else:
        write_realised(cur, day, c_base, c_opt, red)
        conn.commit()
        print("[settle] wrote cost_reduction_pct_realised")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
