"""
fetch_entsoe_prices.py -- day-ahead prices for the MPC forecast-value study.

The site is in Luxembourg, which sits in the DE-LU day-ahead bidding zone
(EIC 10Y1001A1001A82H). Prices are published the day before delivery, so they are
KNOWN at decision time: they need no forecasting, and they are what makes
temporal arbitrage -- and therefore forecast quality -- worth anything.

SECURITY: the API token is read from the ENTSOE_TOKEN environment variable and is
never written to disk. This repository is public; do not hard-code it, and do not
commit the token in any script, notebook or config.

    export ENTSOE_TOKEN=...
    python -m scripts.fetch_entsoe_prices

Writes Data/entsoe_dayahead_DE_LU.csv with a 15-min UTC+01 index matching the
site data (hourly prices are forward-filled onto the 15-min grid, which is the
correct settlement convention: the hourly price applies to each quarter within).
"""
from __future__ import annotations
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

TOKEN = os.environ.get("ENTSOE_TOKEN", "").strip()
DOMAIN = os.environ.get("ENTSOE_DOMAIN", "10Y1001A1001A82H")   # DE-LU bidding zone
START = os.environ.get("PRICE_START", "202512250000")           # UTC, yyyymmddHHMM
END = os.environ.get("PRICE_END", "202601190000")
OUT = os.environ.get("OUT", "Data/entsoe_dayahead_DE_LU.csv")
# CORRECTED 2026-09-16: this comment used to say web-api.tp.entsoe.eu returned
# 404 and transparency.entsoe.eu worked. Both halves were wrong. The 404 came
# from the firewall, not from ENTSO-E, and transparency.entsoe.eu/api serves
# the web APPLICATION -- it answers 200 with an HTML page that parses to zero
# rows, which is worse than an error because it looks like an empty market.
URL = os.environ.get("ENTSOE_URL", "https://web-api.tp.entsoe.eu/api")


def fetch(start, end):
    params = dict(securityToken=TOKEN, documentType="A44",
                  in_Domain=DOMAIN, out_Domain=DOMAIN,
                  periodStart=start, periodEnd=end)
    r = requests.get(URL, params=params, timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"[entsoe] HTTP {r.status_code}: {r.text[:400]}")
    return r.text


def parse(xml_text):
    """Parse a Publication_MarketDocument into (timestamp, EUR/MWh) rows."""
    ns = {"n": xml_text.split('xmlns="')[1].split('"')[0]} if 'xmlns="' in xml_text else {}
    root = ET.fromstring(xml_text)

    def tag(e):
        return e.tag.split("}")[-1]

    rows = []
    for ts in root.iter():
        if tag(ts) != "TimeSeries":
            continue
        for period in [c for c in ts if tag(c) == "Period"]:
            start = res = None
            for c in period:
                if tag(c) == "timeInterval":
                    for cc in c:
                        if tag(cc) == "start":
                            start = pd.Timestamp(cc.text)
                elif tag(c) == "resolution":
                    res = c.text
            if start is None:
                continue
            step = pd.Timedelta(minutes=60 if res in (None, "PT60M") else
                                (15 if res == "PT15M" else 30))
            for pt in [c for c in period if tag(c) == "Point"]:
                pos = price = None
                for cc in pt:
                    if tag(cc) == "position":
                        pos = int(cc.text)
                    elif tag(cc) in ("price.amount", "priceAmount"):
                        price = float(cc.text)
                if pos is not None and price is not None:
                    rows.append((start + (pos - 1) * step, price))
    return pd.DataFrame(rows, columns=["timestamp_utc", "price_eur_mwh"])


def main():
    if not TOKEN:
        raise SystemExit("[entsoe] set ENTSOE_TOKEN in the environment "
                         "(never hard-code it: this repo is public)")
    print(f"[entsoe] domain={DOMAIN} {START} -> {END}")
    df = parse(fetch(START, END))
    if df.empty:
        raise SystemExit("[entsoe] no price points parsed; check dates/domain")
    df = df.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # onto the site's 15-min grid, in its local offset
    idx = pd.date_range(df.timestamp_utc.iloc[0], df.timestamp_utc.iloc[-1],
                        freq="15min", tz="UTC")
    s = df.set_index("timestamp_utc").price_eur_mwh.reindex(idx).ffill()
    out = pd.DataFrame({"timestamp": s.index.tz_convert("Etc/GMT-1"),
                        "price_eur_mwh": s.to_numpy(),
                        "price_eur_kwh": s.to_numpy() / 1000.0})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.describe().T[["mean", "min", "max"]].round(4).to_string())
    print(f"[entsoe] {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
