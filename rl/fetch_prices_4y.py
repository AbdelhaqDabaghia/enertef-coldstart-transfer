"""
fetch_prices_4y.py -- four years of DE-LU day-ahead prices, in yearly chunks.

The site sits in the DE-LU bidding zone. Day-ahead prices are published the day
before delivery, so at decision time they are KNOWN, not forecast. They are
what makes temporal arbitrage worth anything, and therefore the only input the
controller can exploit with certainty.

SECURITY: the token is read from ENTSOE_TOKEN and never written to disk. This
repository is public. Do not hard-code it, do not commit it.

    export ENTSOE_TOKEN=...
    python -m rl.fetch_prices_4y

Writes Data/entsoe_dayahead_DE_LU_4y.csv on a 15-min UTC grid. Hourly prices
are forward-filled onto the quarter-hour grid, which is the settlement
convention: the hourly price applies to each quarter inside it.

ENTSO-E refuses ranges longer than about a year, so the window is requested one
year at a time and the pieces concatenated. A chunk that fails is reported and
skipped rather than aborting the run -- four years with one bad month is still
far better than nothing -- but the summary at the end says exactly what is
missing, so a partial pull can never be mistaken for a complete one.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_entsoe_prices import fetch, parse  # noqa: E402

OUT = os.environ.get("OUT", "Data/entsoe_dayahead_DE_LU_4y.csv")
YEARS = int(os.environ.get("YEARS", "4"))
END = pd.Timestamp(os.environ.get("PRICE_END", "2026-01-19"), tz="UTC")


def main():
    if not os.environ.get("ENTSOE_TOKEN", "").strip():
        raise SystemExit("[prices] ENTSOE_TOKEN is not set. export it first.")

    start = END - pd.DateOffset(years=YEARS)
    edges = pd.date_range(start, END, freq="365D").tolist()
    if edges[-1] < END:
        edges.append(END)

    frames, missing = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        tag = "%s..%s" % (a.date(), b.date())
        try:
            rows = parse(fetch(a.strftime("%Y%m%d%H%M"), b.strftime("%Y%m%d%H%M")))
            df = pd.DataFrame(rows, columns=["timestamp", "price"])
            frames.append(df)
            print("[prices] %s  %6d rows" % (tag, len(df)), flush=True)
        except SystemExit as exc:
            missing.append(tag)
            print("[prices] %s  FAILED: %s" % (tag, exc), flush=True)
        time.sleep(2)          # be a good citizen; the 2026-09-10 block was ours

    if not frames:
        raise SystemExit("[prices] nothing fetched; check the token and the URL")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")

    # hourly -> 15 min, forward-filled inside each hour (settlement convention)
    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(),
                         freq="15min", tz="UTC")
    out = (df.set_index("timestamp")["price"].reindex(grid).ffill(limit=3)
             .rename("price").rename_axis("timestamp").reset_index())
    out = out.dropna()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    days = len(out) / 96.0
    print("\n[prices] %s  %d rows = %.1f days = %.2f years"
          % (OUT, len(out), days, days / 365.0))
    print("[prices] %s -> %s | EUR/MWh mean %.1f min %.1f max %.1f"
          % (out["timestamp"].min().date(), out["timestamp"].max().date(),
             out["price"].mean(), out["price"].min(), out["price"].max()))
    if missing:
        print("[prices] INCOMPLETE -- these chunks failed: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
