"""
e26_censoring_probe.py -- is the EV target censored by its own controller, and
is the PV target not?

THE HYPOTHESIS. The MPC commands EMOB1 and EMOB2 and nothing else: the only
decision variables are u1 and u2, and the only MQTT topics published are
enertef/grid/emob{1,2}/command. PV enters the programme as a parameter. So the
EV forecaster now adapts on data its own decisions shaped,

    y_t = max(y0_t + u_t, 0)

while the PV forecaster, on the same site, the same pipeline, the same nightly
schedule and the same 90-day window, adapts on data nothing touched. PV is
therefore a CONTROL GROUP for the censoring hypothesis, and few sites can
produce one: the value-oriented forecasting literature works on renewables,
where the controlled case does not exist.

WHAT THIS SCRIPT MEASURES. For every dispatched cycle since the actuation
boundary it recovers the uncontrolled baseline where that is possible and marks
it where it is not:

    y_t > 0            -> y0_t = y_t - u_t          exactly recoverable
    y_t = 0 and u_t<0  -> y0_t <= -u_t              only an upper bound: CENSORED

and reports the censored fraction, the energy it hides, and the same quantities
for PV, which should be identically zero. A non-zero PV figure would mean the
premise is wrong and the asymmetry is not what we think.

It also reports how much of the EV training window is now controlled data,
which is what decides when the contamination becomes large enough to matter for
the nightly adaptation.

    bash scripts/pg_tunnel.sh
    export PG_HOST=127.0.0.1 PG_PORT=15432 PG_DB=... PG_USER=... PG_PASSWORD=...
    python scripts/e26_censoring_probe.py

Writes Data/results/e26_censoring_probe.csv. Read-only: no write to any table.
"""
from __future__ import annotations

import os
import ssl
import sys

import numpy as np
import pandas as pd

# The instant the control record became reliable -- see the settlement section.
# Before it, rows carry status 'pending' even for setpoints that were dispatched,
# so they cannot be used to reconstruct anything.
BOUNDARY = "2026-09-16 06:33:52+00"
OUT = os.environ.get("OUT", "Data/results/e26_censoring_probe.csv")

SQL_SETPOINTS = """
    SELECT target_time, asset_id, SUM(setpoint_kw) AS u
      FROM planning.mpc_setpoints
     WHERE status = 'sent'
       AND target_time = computed_at
       AND target_time >= %s
     GROUP BY target_time, asset_id
     ORDER BY target_time
"""

SQL_TELEMETRY = """
    SELECT timestamp, asset_id, power_kw_avg AS value_kw
      FROM historical.telemetry_15min
     WHERE timestamp >= %s
       AND asset_id IN ('EMOB1', 'EMOB2', 'PV')
     ORDER BY timestamp
"""


def connect():
    import pg8000.dbapi
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return pg8000.dbapi.connect(
        host=os.environ.get("PG_HOST", "127.0.0.1"),
        port=int(os.environ.get("PG_PORT", "15432")),
        database=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"], ssl_context=ctx)


def fetch(cur, sql, *args):
    cur.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def main():
    for v in ("PG_DB", "PG_USER", "PG_PASSWORD"):
        if not os.environ.get(v):
            raise SystemExit("[e26] %s is not set" % v)

    c = connect()
    cur = c.cursor()
    sp = fetch(cur, SQL_SETPOINTS, BOUNDARY)
    tm = fetch(cur, SQL_TELEMETRY, BOUNDARY)
    cur.close()
    c.close()

    print("[e26] boundary %s" % BOUNDARY)
    print("[e26] dispatched setpoint rows : %d" % len(sp))
    print("[e26] telemetry rows           : %d" % len(tm))
    if sp.empty or tm.empty:
        raise SystemExit("[e26] nothing to measure yet; telemetry lags ~2 days")

    sp["target_time"] = pd.to_datetime(sp["target_time"], utc=True)
    # Setpoint rows carry the cycle's wall-clock seconds (06:33:52) while the
    # telemetry sits on a clean quarter-hour grid. Joining them as-is finds no
    # match at all -- the failure mode that produced two wrong versions of
    # settle_kpis. Floor to the slot the setpoint acts on, and report both the
    # floor and the ceiling so the choice is visible rather than assumed.
    sp["slot_floor"] = sp["target_time"].dt.floor("15min")
    sp["slot_ceil"] = sp["target_time"].dt.ceil("15min")
    tm["timestamp"] = pd.to_datetime(tm["timestamp"], utc=True)
    print("[e26] setpoints %s -> %s" % (sp.target_time.min(), sp.target_time.max()))
    print("[e26] telemetry %s -> %s" % (tm.timestamp.min(), tm.timestamp.max()))

    rows = []
    for asset in ("EMOB1", "EMOB2", "PV"):
        t = tm[tm.asset_id == asset].set_index("timestamp")["value_kw"].astype(float)
        sa = sp[sp.asset_id == asset]
        u = pd.Series(dtype=float)
        which = "none"
        if len(sa):
            for col, name in (("slot_ceil", "ceil"), ("slot_floor", "floor")):
                cand = sa.groupby(col)["u"].sum().astype(float)
                hits = cand.index.isin(t.index).sum()
                if hits > len(u.reindex(t.index).dropna() if len(u) else []):
                    u, which = cand, "%s (%d/%d aligned)" % (name, hits, len(cand))
            print("  %-6s alignement retenu : %s" % (asset, which))
        if t.empty:
            print("  %-6s no telemetry" % asset)
            continue

        # align on the telemetry grid; a slot with no setpoint was not commanded
        uu = u.reindex(t.index).fillna(0.0) if len(u) else pd.Series(0.0, index=t.index)
        commanded = int((uu != 0).sum())
        # censored: realised at the floor while a reduction was commanded
        censored = (t <= 0.05) & (uu < 0)
        recoverable = (t > 0.05) & (uu != 0)
        hidden_kwh = float((-uu[censored]).sum() * 0.25)

        rows.append(dict(
            asset=asset, slots=len(t), commanded=commanded,
            commanded_pct=round(100 * commanded / len(t), 2),
            censored=int(censored.sum()),
            censored_pct_of_commanded=(round(100 * censored.sum() / commanded, 2)
                                       if commanded else 0.0),
            recoverable=int(recoverable.sum()),
            hidden_kwh=round(hidden_kwh, 2),
            mean_realised_kw=round(float(t.mean()), 2),
        ))
        print("  %-6s %5d slots | commanded %5d (%5.2f %%) | censored %4d | "
              "hidden %7.1f kWh"
              % (asset, len(t), commanded, 100 * commanded / len(t),
                 int(censored.sum()), hidden_kwh))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== the asymmetry ===")
    ev = df[df.asset.isin(("EMOB1", "EMOB2"))]
    pv = df[df.asset == "PV"]
    if not ev.empty:
        print("  EV : %d commanded slots, %d censored, %.1f kWh hidden"
              % (ev.commanded.sum(), ev.censored.sum(), ev.hidden_kwh.sum()))
    if not pv.empty:
        n = int(pv.commanded.iloc[0])
        print("  PV : %d commanded slots, %d censored  %s"
              % (n, int(pv.censored.iloc[0]),
                 "<- as expected: the controller does not touch PV"
                 if n == 0 else "<- UNEXPECTED, the premise needs checking"))
    print("\n[e26] wrote %s" % OUT)


if __name__ == "__main__":
    main()
