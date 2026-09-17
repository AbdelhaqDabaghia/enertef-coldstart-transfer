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

ENTSO-E refuses ranges longer than about a year, and for anything large it
answers with a ZIP of XML documents rather than one XML document -- a year of
day-ahead prices comes back as `PK...`, which an XML parser rejects at the
first byte. Both shapes are handled here, and the window is requested in
180-day slices to keep each response small.

A slice that fails is reported and skipped rather than aborting the run -- four
years with one bad month beats nothing -- but the summary at the end names
exactly what is missing, so a partial pull can never be mistaken for a complete
one. When a slice comes back unparseable, its raw body is written next to the
output so the failure can be read instead of guessed at.
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_entsoe_prices import parse  # noqa: E402

OUT = os.environ.get("OUT", "Data/entsoe_dayahead_DE_LU_4y.csv")
YEARS = int(os.environ.get("YEARS", "4"))
END = pd.Timestamp(os.environ.get("PRICE_END", "2026-01-19"), tz="UTC")
DOMAIN = os.environ.get("ENTSOE_DOMAIN", "10Y1001A1001A82H")
# web-api.tp.entsoe.eu is the API. transparency.entsoe.eu/api serves the web
# APPLICATION: it answers 200 with an HTML page, which parses to zero rows and
# would otherwise be indistinguishable from an empty market. The earlier note
# in scripts/fetch_entsoe_prices.py claiming the opposite was wrong -- the 404
# seen then came from the firewall, and the dead token was returning 401.
URL = os.environ.get("ENTSOE_URL", "https://web-api.tp.entsoe.eu/api")
CHUNK_DAYS = int(os.environ.get("CHUNK_DAYS", "180"))


class SliceFailed(Exception):
    pass


def fetch_rows(start, end, tag):
    """One slice -> a DataFrame of (timestamp, price).

    scripts.fetch_entsoe_prices.parse() returns a DataFrame, not a list of
    tuples, and its columns are timestamp_utc / price_eur_mwh. Iterating it
    yields the column NAMES, which is what produced the
    "Shape of passed values is (2, 1)" failure. Concatenate, then rename.
    """
    params = dict(securityToken=os.environ["ENTSOE_TOKEN"].strip(),
                  documentType="A44", in_Domain=DOMAIN, out_Domain=DOMAIN,
                  periodStart=start, periodEnd=end)
    r = requests.get(URL, params=params, timeout=180)
    if r.status_code != 200:
        raise SliceFailed("HTTP %d: %s" % (r.status_code, r.text[:300]))

    body = r.content
    if body[:100].lstrip()[:9].lower() == b"<!doctype" or b"<html" in body[:400].lower():
        raise SliceFailed(
            "got an HTML page, not the API. ENTSOE_URL is %s -- the API lives "
            "at https://web-api.tp.entsoe.eu/api" % URL)
    docs = []
    if body[:2] == b"PK":                       # a ZIP of XML documents
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for name in z.namelist():
                docs.append(z.read(name).decode("utf-8", "replace"))
    else:
        docs.append(body.decode("utf-8", "replace"))

    frames = []
    for doc in docs:
        if "Acknowledgement_MarketDocument" in doc:
            reason = doc.split("<text>")[-1].split("</text>")[0] if "<text>" in doc else ""
            raise SliceFailed("ENTSO-E refused the slice: %s" % reason[:200])
        try:
            frames.append(parse(doc))
        except Exception as exc:
            dump = os.path.join(os.path.dirname(OUT) or ".",
                                "_entsoe_raw_%s.txt" % tag.replace("..", "_"))
            os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
            with open(dump, "w", encoding="utf-8") as fh:
                fh.write(doc[:200000])
            raise SliceFailed("unparseable (%s); raw body written to %s"
                              % (exc, dump))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise SliceFailed("no TimeSeries in the response")
    return (df.rename(columns={"timestamp_utc": "timestamp",
                               "price_eur_mwh": "price"})[["timestamp", "price"]])


def main():
    if not os.environ.get("ENTSOE_TOKEN", "").strip():
        raise SystemExit("[prices] ENTSOE_TOKEN is not set. export it first.")

    start = END - pd.DateOffset(years=YEARS)
    edges = pd.date_range(start, END, freq="%dD" % CHUNK_DAYS).tolist()
    if edges[-1] < END:
        edges.append(END)

    frames, missing = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        tag = "%s..%s" % (a.date(), b.date())
        try:
            df = fetch_rows(a.strftime("%Y%m%d%H%M"),
                            b.strftime("%Y%m%d%H%M"), tag)
            frames.append(df)
            print("[prices] %s  %6d rows" % (tag, len(df)), flush=True)
        except Exception as exc:
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
