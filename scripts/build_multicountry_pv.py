"""
Build multi-COUNTRY PV targets (reviewer point #3: geographic diversity) from
OPSD time_series national solar generation, in the winning PER-UNIT representation
(divide by national capacity -> dimensionless). Weather from each country's
centroid (Open-Meteo) so the WEATHER distance now varies across targets.
Span 2018-2019, resampled to the 15-min grid the PV model expects.

Writes Data/multicountry/<CC>_features.csv (11 PV features, pv_kw = per-unit gen).
"""
import numpy as np, pandas as pd, requests
from coldstart_transfer.pv import engineer_pv_features, PV_WEATHER

CENTROIDS = {  # sunny south -> cloudy north/Baltic
    "PT": (39.5, -8.0), "ES": (40.0, -3.7), "IT": (42.5, 12.5), "GR": (39.0, 22.0),
    "FR": (46.5, 2.5), "DE": (51.0, 10.0), "NL": (52.1, 5.3), "CZ": (49.8, 15.5),
    "DK": (56.0, 9.5), "EE": (58.7, 25.0),
}
START, END = "2018-01-01", "2019-12-31"

def fetch_weather(lat, lon):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "hourly": ",".join(PV_WEATHER), "timezone": "UTC"}, timeout=180)
    r.raise_for_status(); h = r.json()["hourly"]
    return pd.DataFrame({"timestamp": pd.to_datetime(h["time"], utc=True), **{v: h[v] for v in PV_WEATHER}})

def main():
    ts = pd.read_csv("Data/multicountry/opsd_timeseries_60min.csv", low_memory=False)
    ts["t"] = pd.to_datetime(ts["utc_timestamp"], utc=True)
    ts = ts[(ts.t >= START) & (ts.t <= END)]
    grid = pd.date_range(START, END, freq="15min", tz="UTC")
    for cc, (lat, lon) in CENTROIDS.items():
        col = f"{cc}_solar_generation_actual"
        if col not in ts.columns or ts[col].notna().sum() < 200*24:
            print(f"[skip] {cc}: no/short data"); continue
        s = ts.set_index("t")[col].dropna()
        cap = float(np.quantile(s, 0.999)); pu = (s / cap).clip(0, 1.2)
        pu15 = pu.reindex(pu.index.union(grid)).interpolate("time").reindex(grid).ffill().bfill()
        wx = fetch_weather(lat, lon).set_index("timestamp").sort_index()
        wx15 = wx.reindex(wx.index.union(grid)).interpolate("time").reindex(grid).ffill().bfill()
        raw = pd.DataFrame({"timestamp": grid, "pv_kw": pu15.to_numpy()})
        for c in PV_WEATHER: raw[c] = wx15[c].to_numpy()
        feat = engineer_pv_features(raw)
        feat.to_csv(f"Data/multicountry/{cc}_features.csv", index=False)
        print(f"[{cc}] cap={cap:.0f}MW  {len(feat)} rows  pu_mean={feat.pv_kw.mean():.3f}", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
