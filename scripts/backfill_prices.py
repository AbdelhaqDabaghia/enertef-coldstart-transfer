"""
backfill_prices.py -- refill contextual.prices for the days the fetcher missed.

WHY THIS EXISTS. The ENTSO-E token was rotated at ENTSO-E on 2026-09-16 but not
in Secrets Manager, where enertef-price-fetcher reads it. The fetcher answered
401 every day from the 16th; contextual.prices froze; the controller fell back
to a flat 100 EUR/MWh; and under a flat price an energy-conserving shift cannot
change the bill, so every cycle settled at exactly 0.00 EUR and the gate read
FAIL. Six days of the first real settlement window, lost to a no-op that
reported itself as a scheduled run.

The fetcher only knows how to fetch "tomorrow", so those days cannot be
recovered by re-running it. It is also a container-image Lambda, so patching
it means rebuilding and pushing an image -- too heavy for a one-off. This
script does the same fetch and the SAME upsert, from a workstation, through the
bastion tunnel.

    bash scripts/pg_tunnel.sh
    export PG_HOST=127.0.0.1 PG_PORT=15432 PG_DB=... PG_USER=... PG_PASSWORD=...
    export ENTSOE_TOKEN=...
    python scripts/backfill_prices.py --start 2026-09-16 --end 2026-09-23

Credentials come from the environment and are never written anywhere. This
repository is public.

The upsert matches the Lambda's exactly -- ON CONFLICT (timestamp) DO UPDATE,
source 'entsoe_dayahead' -- so a day already present is overwritten with the
same values and a day the Lambda later re-fetches overwrites this. Idempotent
in both directions.

It refuses to write if the fetch returned fewer than 20 hours for any day, so a
partial answer cannot be mistaken for a complete one.
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.fetch_prices_4y import fetch_rows  # noqa: E402  (ZIP/XML/HTML aware)


def upsert(df):
    import pg8000.dbapi
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    c = pg8000.dbapi.connect(
        host=os.environ.get("PG_HOST", "127.0.0.1"),
        port=int(os.environ.get("PG_PORT", "15432")),
        database=os.environ["PG_DB"], user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"], ssl_context=ctx)
    cur = c.cursor()
    n = 0
    for ts, price in zip(df["timestamp"], df["price"]):
        cur.execute("""
            INSERT INTO contextual.prices (timestamp, price_eur_mwh, source)
            VALUES (%s, %s, 'entsoe_dayahead')
            ON CONFLICT (timestamp) DO UPDATE
            SET price_eur_mwh = EXCLUDED.price_eur_mwh,
                source = EXCLUDED.source,
                created_at = NOW()
        """, (ts.to_pydatetime(), float(price)))
        n += 1
    c.commit()
    cur.close()
    c.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first day, UTC, inclusive")
    ap.add_argument("--end", required=True, help="day AFTER the last one, UTC")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for v in ("ENTSOE_TOKEN", "PG_DB", "PG_USER", "PG_PASSWORD"):
        if not os.environ.get(v):
            raise SystemExit("[backfill] %s is not set" % v)

    a = pd.Timestamp(args.start, tz="UTC")
    b = pd.Timestamp(args.end, tz="UTC")
    df = fetch_rows(a.strftime("%Y%m%d%H%M"), b.strftime("%Y%m%d%H%M"),
                    "%s..%s" % (a.date(), b.date()))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")

    # hourly rows are what ENTSO-E returns; the controller reads the hourly
    # price for each quarter, and the Lambda stores hourly too -- same grain.
    per_day = df.groupby(df["timestamp"].dt.date).size()
    print("[backfill] %d rows, %s -> %s | EUR/MWh mean %.1f min %.1f max %.1f"
          % (len(df), df["timestamp"].min(), df["timestamp"].max(),
             df["price"].mean(), df["price"].min(), df["price"].max()))
    for d, n in per_day.items():
        print("   %s  %2d h" % (d, n))
    short = per_day[per_day < 20]
    if len(short):
        raise SystemExit("[backfill] REFUSING: incomplete days %s -- a partial "
                         "fill would look complete to the controller"
                         % ", ".join(str(d) for d in short.index))

    if args.dry_run:
        print("[backfill] dry run, nothing written")
        return
    n = upsert(df)
    print("[backfill] upserted %d rows into contextual.prices" % n)


if __name__ == "__main__":
    main()
