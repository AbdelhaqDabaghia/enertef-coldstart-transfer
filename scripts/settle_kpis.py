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
hard-code them.

The database is not reachable directly from a workstation: TCP to 5432 connects
but the PostgreSQL SSLRequest gets no reply -- deep packet inspection dropping
the payload, measured 2026-09-16. Open scripts/pg_tunnel.sh first, which
forwards through enertef-bastion (same VPC, SSH passes), then point PG_HOST at
127.0.0.1. Running inside the VPC also works and needs no tunnel.

A SETTLEABLE DAY MUST SATISFY THREE THINGS, and today usually will not:
  * it is at or after the actuation boundary (2026-09-16 06:19 UTC) -- before
    that no setpoint reached a charger;
  * its telemetry has been ingested, which happens the NEXT morning: Leneda
    publishes with a lag of about two hours at the end of the day and nothing
    at all for the day in progress (measured 2026-09-16);
  * day-ahead prices cover it, which they do a day ahead.
So the earliest a day can be settled is the morning after it ends.
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

# contextual.prices, NOT enertef.market_prices. The latter is defined in
# schema.sql and written by nothing -- the same trap as enertef.mpc_runs. This
# query was originally written from the schema file rather than from what the
# runner actually reads, and failed with 'relation "market_prices" does not
# exist' the first time it touched a real database. Read the code that runs, not
# the schema that documents.
#
# There is no bidding_zone column: the price_fetcher writes a single zone
# (DE_LU, to which Luxembourg is coupled for day-ahead).
SQL_PRICE = """
    SELECT timestamp, price_eur_mwh
      FROM contextual.prices
     WHERE timestamp::date = %s
     ORDER BY timestamp
"""

# WHAT THE CONTROLLER ACTUALLY DISPATCHES, and why two earlier versions of this
# query were wrong.
#
# Every 15-minute cycle solves a 96-step horizon and writes all 96 steps per
# charger. A single day therefore holds ~8500 rows per charger from ~150 cycles,
# and any given target_time is written by roughly a hundred successive cycles.
#
#   v1 selected raw rows ordered by target_time. With per-charger rows that
#      returned a 192+ element array for a 96-step horizon and silently settled
#      interleaved fragments. No error: both arrays were plausible lengths.
#   v2 grouped by target_time and SUMmed. That summed ~100 revisions of the same
#      instant and would have overstated the deviation by about two orders of
#      magnitude. Also silent.
#
# The controller publishes only u[0] -- the FIRST step of each cycle. Verified
# in the database: every row with status='sent' has target_time == computed_at
# exactly (delta 0 min). So the trajectory the site actually received is the
# SEQUENCE OF DISPATCHED FIRST STEPS, one per cycle, not a 96-step plan.
#
# Settling the plan would measure a controller that was never run. Settling the
# dispatched steps measures the one that was.
#
# Site-level deviation is u1 + u2 summed ACROSS CHARGERS at the same instant --
# never across cycles.
SQL_SETPOINTS = """
    SELECT target_time,
           SUM(setpoint_kw)                        AS u_site,
           COUNT(*)                                AS n_chargers
      FROM planning.mpc_setpoints
     WHERE service_id = 1
       AND target_time::date = %s
       AND status = 'sent'
       AND target_time = computed_at
     GROUP BY target_time
     ORDER BY target_time
"""

# How many cycles ran that day at all, so a dispatch gap is visible rather than
# silently shortening the settled period.
SQL_CYCLES = """
    SELECT COUNT(DISTINCT computed_at) AS cycles,
           COUNT(DISTINCT computed_at) FILTER (WHERE status = 'sent') AS dispatched
      FROM planning.mpc_setpoints
     WHERE service_id = 1 AND computed_at::date = %s
"""

# TWO BOUNDARIES, and settlement must use the later one.
#
#   06:19:31 UTC  the MQTT publish first SUCCEEDED (runner logs). Before this,
#                 400 consecutive failures over a fortnight: no setpoint ever
#                 reached a charger, so any earlier "saving" is counterfactual.
#   06:33:52 UTC  image sha256:3b4021f5 shipped, bringing d395ac0 (per-charger
#                 actuation status) and f113630 (per-charger rows).
#
# Between them the controller WAS actuating -- the logs prove it -- but the
# database cannot say so. Those rows carry status='pending' and exist for EMOB1
# only, because the code that records dispatch had not shipped yet. Verified:
#
#   06:19:31  EMOB1  pending     <- dispatched in fact, DB cannot tell
#   06:19:46  EMOB1  pending     <- dispatched in fact, DB cannot tell
#   06:33:52  EMOB1  sent    } first cycle the database can be trusted about
#   06:33:52  EMOB2  sent    }
#
# Settlement therefore keys on 06:33:52 -- not when control began, but when the
# RECORD of control became reliable. Admitting the earlier window would settle
# genuinely dispatched steps as u = 0, understating the controller rather than
# overstating it, but wrong either way and silently so.
ACTUATION_BOUNDARY = dt.datetime(2026, 9, 16, 6, 33, 52, tzinfo=dt.timezone.utc)
FIRST_ACTUATION_OBSERVED = dt.datetime(2026, 9, 16, 6, 19, 31,
                                       tzinfo=dt.timezone.utc)

SQL_PLANNED = """
    SELECT measured_value
      FROM historical.kpi_validation
     WHERE service_id = 1 AND kpi_name = 'cost_reduction_pct_planned'
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
    # Through scripts/pg_tunnel.sh the connection is to 127.0.0.1 while the
    # server presents the RDS certificate, so hostname verification necessarily
    # fails. Relax it ONLY for the loopback case: the SSH tunnel already
    # authenticates the far end and encrypts the hop, so TLS here is defence in
    # depth rather than the primary control. A direct connection keeps full
    # verification, which is what matters -- this must not become a blanket
    # "verify nothing" that quietly applies in production too.
    if os.environ["PG_HOST"] in ("127.0.0.1", "localhost", "::1"):
        print("[settle] loopback host: TLS hostname check relaxed (tunnelled "
              "connection; the SSH hop authenticates the endpoint)")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
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
        pairs = sorted(by_asset.get(name, []))
        return [t for t, _ in pairs], np.array([v for _, v in pairs])

    ts_ev, ev1 = series("EMOB1")
    _, ev2 = series("EMOB2")
    ts_pv, pv_real = series("PV")
    ev_real = ev1 + ev2
    # Timestamps are the alignment key from here on, not positions: the
    # dispatched setpoints are sparse and will not line up by index.
    telemetry_times = ts_ev

    cur.execute(SQL_PRICE, (day,))
    price = np.array([float(p) for _, p in cur.fetchall()])

    n = min(len(ev_real), len(pv_real), len(price))
    if n == 0:
        raise SystemExit("[settle] missing telemetry or prices for %s" % day)
    if n < 96:
        print("[settle] WARNING: only %d of 96 steps available; the day is "
              "incomplete and the figure below is partial" % n)

    # Refuse to settle a period in which nothing was dispatched.
    if dt.datetime.combine(day, dt.time(23, 59), dt.timezone.utc) < ACTUATION_BOUNDARY:
        raise SystemExit(
            "[settle] %s is entirely before the actuation boundary %s.\n"
            "  No setpoint reached a charger before that instant (400 "
            "consecutive MQTT publish failures), so settling this day would\n"
            "  produce a COUNTERFACTUAL -- what the site would have saved had "
            "the plan been applied -- labelled as realised.\n"
            "  That is the error this script exists to prevent. Use the "
            "counterfactual figures in e18 for earlier periods, and say so."
            % (day, ACTUATION_BOUNDARY.isoformat()))

    # The issued plan, aggregated to site level. One row per charger per step
    # since f113630, so SUM over chargers; the grid sees u1 + u2.
    cur.execute(SQL_SETPOINTS, (day,))
    sp = cur.fetchall()
    if not sp:
        raise SystemExit(
            "[settle] no setpoints in planning.mpc_setpoints for %s.\n"
            "  Without the plan that was issued there is nothing to settle."
            % day)

    # Map the dispatched deviations onto the telemetry timeline BY TIMESTAMP.
    # They are sparse -- one per cycle that actually published -- so a step with
    # no dispatched setpoint received no control and takes u = 0. Aligning by
    # position instead would slide the whole trajectory, which is the error the
    # two earlier versions of this query made in different ways.
    dispatched = {r[0]: float(r[1]) for r in sp}
    chargers = {int(r[2]) for r in sp}
    u_plan = np.array([dispatched.get(t, 0.0) for t in telemetry_times])
    n_controlled = int(np.count_nonzero(u_plan))

    cur.execute(SQL_CYCLES, (day,))
    n_cycles, n_dispatched_cycles = cur.fetchone()

    print("[settle] cycles that ran    : %d, of which dispatched: %d"
          % (n_cycles or 0, n_dispatched_cycles or 0))
    print("[settle] controlled steps   : %d of %d telemetry steps"
          % (n_controlled, len(telemetry_times)))
    if chargers and chargers != {2}:
        print("[settle] WARNING: expected 2 chargers per dispatched step; saw "
              "%s. Before svc1-runner f113630 a single row carried the site "
              "total, so {1} means this day predates that fix." % sorted(chargers))
    if n_controlled < len(telemetry_times):
        gap = len(telemetry_times) - n_controlled
        print("[settle] NOTE: %d of %d steps had no dispatched setpoint and are "
              "settled with u = 0 (no control). The site was unmanaged for "
              "those steps, which is the honest treatment -- but it means the "
              "figure below covers a partially controlled day."
              % (gap, len(telemetry_times)))

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
