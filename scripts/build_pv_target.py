"""
build_pv_target.py -- build the Konstanz PV target for the cold-start replication.

Source: OPSD household_data 15-min, system DE_KN_residential3_pv (cumulative kWh
counter). We convert to power, clean counter resets, align to a 15-min grid, then
re-fetch Open-Meteo weather for Konstanz and assemble the 11 PV features.

Output: Data/pv_target/konstanz_pv_features.csv (columns == PV_SEQ_FEATURES).
"""
import sys
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from coldstart_transfer.pv import engineer_pv_features, PV_WEATHER

LAT, LON = 47.6779, 9.1732           # Konstanz, DE
SYS = "DE_KN_residential3_pv"
CAP_KW = 15.0                        # physical clip for residential resets/spikes


def load_power():
    df = pd.read_csv("Data/pv_target/opsd_household_15min.csv", low_memory=False,
                     usecols=["utc_timestamp", SYS])
    df["ts"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df[["ts", SYS]].dropna().sort_values("ts")
    # clean 15-min grid over the covered span
    grid = pd.date_range(df.ts.min(), df.ts.max(), freq="15min", tz="UTC")
    s = df.set_index("ts")[SYS].reindex(grid).interpolate(limit=8)  # bridge <=2h gaps
    # cumulative kWh -> power kW = dE * 4 (per 15 min); reset/spike cleanup
    p = (s.diff() * 4.0).clip(lower=0.0, upper=CAP_KW)
    out = pd.DataFrame({"timestamp": grid, "pv_kw": p.to_numpy()}).dropna().reset_index(drop=True)
    print(f"[pv] {SYS}: {len(out)} steps [{out.timestamp.iloc[0]} .. {out.timestamp.iloc[-1]}]  "
          f"pv_kw max={out.pv_kw.max():.2f} mean={out.pv_kw.mean():.3f}")
    return out


def fetch_weather(start, end):
    print(f"[wx] Open-Meteo {start.date()}..{end.date()} @({LAT},{LON})")
    hourly = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
              "cloud_cover", "temperature_2m", "wind_speed_10m"]
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON, "start_date": str(start.date()),
        "end_date": str(end.date()), "hourly": ",".join(hourly), "timezone": "UTC"},
        timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    wx = pd.DataFrame({"timestamp": pd.to_datetime(h["time"], utc=True),
                       **{v: h[v] for v in hourly}})
    print(f"[wx] {len(wx)} hourly rows")
    return wx


def main():
    ev = load_power()
    wx = fetch_weather(ev.timestamp.iloc[0] - pd.Timedelta(days=1),
                       ev.timestamp.iloc[-1] + pd.Timedelta(days=1)).set_index("timestamp").sort_index()
    grid = pd.DatetimeIndex(ev["timestamp"])
    wx15 = wx.reindex(wx.index.union(grid)).interpolate("time").reindex(grid).ffill().bfill()
    raw = ev.copy()
    for c in PV_WEATHER:
        raw[c] = wx15[c].to_numpy()
    feat = engineer_pv_features(raw)
    feat.to_csv("Data/pv_target/konstanz_pv_features.csv", index=False)
    print(f"[out] wrote Data/pv_target/konstanz_pv_features.csv ({len(feat)} rows, "
          f"{feat.shape[1]} cols)")


if __name__ == "__main__":
    main()
