"""
telemetry_sanity.py -- daily physical checks on the ingested series.

WHY. The POD -> label mapping lives only in the ingest job and the proxy.
historical.telemetry_15min stores a LABEL ('EMOB1', 'EMOB2', 'PV'), not a POD
and not a foreign key, so if a meter were ever wired to the wrong label every
row would be mislabelled and nothing in the database could detect it. Migration
002 updates the identifiers in enertef.metering_points by label precisely
because a positional update would swap all three silently.

Physics can detect what the schema cannot. PV is the only asset that must read
zero at night. Charging demand cannot be negative. A meter that has stopped
reporting looks identical to a genuinely idle site unless someone checks.

Run this after the J-1 ingest, before settlement. It exits non-zero on a failed
check so a scheduler can alert.

    python scripts/telemetry_sanity.py --day 2026-09-14
    python scripts/telemetry_sanity.py --from-csv Data/ECC_master_PV_EMOB1_EMOB2_15min.csv

Credentials come from the environment (PG_HOST, PG_DB, PG_USER, PG_PASSWORD).
Run inside the VPC: the corporate firewall blocks the PostgreSQL wire protocol
from a laptop.
"""
from __future__ import annotations
import os
import sys
import argparse
import datetime as dt
import numpy as np
import pandas as pd

# LOCAL night, conservatively inside darkness year-round at 49.5 N.
#
# The timestamps are UTC and Luxembourg runs UTC+1/+2, so these hours MUST be
# converted before use. Applying them to UTC directly flags 01:00-04:00 UTC,
# which in June is 03:00-06:00 local -- after sunrise. Tested against three
# years of site data, that mistake produces 141 false positives, every one of
# them at 05:45 local in May, June or July, i.e. real generation at dawn.
# Converted to local time the same data gives zero violations and a maximum of
# exactly 0.000 kW, which is also a clean confirmation that the PV label is
# correctly assigned today.
#
# A check that cries wolf is worse than no check: it teaches people to ignore
# the alert that will one day be real.
SITE_TZ = "Europe/Luxembourg"
NIGHT_START, NIGHT_END = 1, 4
PV_NIGHT_TOLERANCE_KW = 1.0     # inverter standby / metering noise
MIN_REPORTING_FRACTION = 0.9    # of 96 expected 15-min steps


class SanityFailure(RuntimeError):
    pass


def check_pv_dark_at_night(ts, pv_kw):
    """The one check that can catch a POD/label swap.

    If PV were mislabelled onto a charging meter, this fires on the first
    night. Nothing else in the stack can see that error."""
    idx = pd.DatetimeIndex(ts)
    # Convert to site-local time first; see the SITE_TZ note above.
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    hours = idx.tz_convert(SITE_TZ).hour
    night = (hours >= NIGHT_START) & (hours < NIGHT_END)
    if not night.any():
        return None
    v = np.asarray(pv_kw, float)[night]
    worst = float(np.nanmax(np.abs(v))) if len(v) else 0.0
    if worst > PV_NIGHT_TOLERANCE_KW:
        return ("PV IS NOT DARK AT NIGHT: %.1f kW between %02d:00 and %02d:00 "
                "%s. PV generation at night is physically impossible, so either "
                "the PV label is attached to the wrong meter -- the failure mode "
                "the schema cannot detect -- or the timestamps are not the UTC "
                "this check assumes."
                % (worst, NIGHT_START, NIGHT_END, SITE_TZ))
    return None


def check_demand_non_negative(name, kw):
    v = np.asarray(kw, float)
    worst = float(np.nanmin(v)) if len(v) else 0.0
    if worst < -PV_NIGHT_TOLERANCE_KW:
        return ("%s REPORTS NEGATIVE DEMAND: %.1f kW. A charging point cannot "
                "export; this is a sign convention error or a swapped series."
                % (name, worst))
    return None


def check_reporting_completeness(name, kw, expected=96):
    v = np.asarray(kw, float)
    present = int(np.sum(~np.isnan(v)))
    if present < expected * MIN_REPORTING_FRACTION:
        return ("%s REPORTED ONLY %d OF %d STEPS. An incomplete day settles to "
                "a misleading KPI, because a partial series looks like a quiet "
                "site." % (name, present, expected))
    return None


def check_not_frozen(name, kw):
    """A stuck meter reports the same value forever and looks like a calm day."""
    v = np.asarray(kw, float)
    v = v[~np.isnan(v)]
    if len(v) >= 8 and float(np.nanstd(v)) < 1e-6:
        return ("%s IS FROZEN at %.2f kW for the whole day. A stuck meter is "
                "indistinguishable from an idle site in every aggregate."
                % (name, float(v[0])))
    return None


def run_checks(ts, series: dict) -> list:
    """series: {'PV': array, 'EMOB1': array, 'EMOB2': array}"""
    problems = []
    if "PV" in series:
        problems.append(check_pv_dark_at_night(ts, series["PV"]))
        problems.append(check_not_frozen("PV", series["PV"]))
        problems.append(check_reporting_completeness("PV", series["PV"]))
    for a in ("EMOB1", "EMOB2"):
        if a in series:
            problems.append(check_demand_non_negative(a, series[a]))
            problems.append(check_not_frozen(a, series[a]))
            problems.append(check_reporting_completeness(a, series[a]))
    return [p for p in problems if p]


SQL = """
    SELECT timestamp, asset_id, power_kw_avg
      FROM historical.telemetry_15min
     WHERE timestamp::date = %s AND source = 'energypark-api'
     ORDER BY timestamp
"""


def from_db(day):
    missing = [v for v in ("PG_HOST", "PG_DB", "PG_USER", "PG_PASSWORD")
               if not os.environ.get(v)]
    if missing:
        raise SystemExit("[sanity] missing environment: %s" % ", ".join(missing))
    import pg8000.dbapi
    import ssl
    conn = pg8000.dbapi.connect(
        host=os.environ["PG_HOST"], port=int(os.environ.get("PG_PORT", "5432")),
        database=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        ssl_context=ssl.create_default_context())
    cur = conn.cursor()
    cur.execute(SQL, (day,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    if not rows:
        raise SystemExit("[sanity] no telemetry for %s" % day)
    df = pd.DataFrame(rows, columns=["timestamp", "asset_id", "power_kw_avg"])
    ts = sorted(df["timestamp"].unique())
    series = {a: g.sort_values("timestamp")["power_kw_avg"].astype(float).to_numpy()
              for a, g in df.groupby("asset_id")}
    return pd.DatetimeIndex(ts), series


def from_csv(path, day=None):
    df = pd.read_csv(path, parse_dates=["Started at"])
    df["t"] = pd.to_datetime(df["Started at"], utc=True)
    if day:
        df = df[df["t"].dt.date == dt.date.fromisoformat(day)]
    if df.empty:
        raise SystemExit("[sanity] no rows for that day in %s" % path)
    return pd.DatetimeIndex(df["t"]), {
        "PV": df["PV_TotalProduction_kW"].astype(float).to_numpy(),
        "EMOB1": df["EMOB1_EV_Consumption_kW"].astype(float).to_numpy(),
        "EMOB2": df["EMOB2_EV_Consumption_kW"].astype(float).to_numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY-MM-DD; default = yesterday")
    ap.add_argument("--from-csv", help="check a CSV instead of the database")
    a = ap.parse_args()
    day = a.day or (dt.date.today() - dt.timedelta(days=1)).isoformat()

    ts, series = (from_csv(a.from_csv, a.day) if a.from_csv else from_db(day))
    print("[sanity] %s | %d steps | assets: %s"
          % (day, len(ts), ", ".join(sorted(series))))

    problems = run_checks(ts, series)
    if not problems:
        print("[sanity] all checks passed")
        return 0
    for p in problems:
        print("[sanity] FAIL: %s" % p)
    return len(problems)


if __name__ == "__main__":
    sys.exit(main())
