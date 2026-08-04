"""
build_lux_source.py -- reconstruct the Luxembourg SOURCE feature set for B1/B5/
retention, since the live DB no longer holds the source-window weather (the DB
was re-imported; weather covers only winter-2024 + recent 2026).

EV target: from the local ECC_master (EV_TotalConsumption_kW = EMOB1+EMOB2 over
the EMOB2-active window) -- this reproduces the frozen scaler exactly (38,883
train rows, ev_kw in [0, 578.4], validated offline).

Weather: RE-FETCHED from Open-Meteo Archive for Copal Dudelange (LAT 49.70673,
LON 6.48714) -- the same provider/coords production used (fetch_weather_2023.py).
Hourly -> interpolated to the 15-min grid. This is an APPROXIMATION of the exact
DB snapshot the scaler saw; documented as a known limitation (like the UK
single-point weather approximation).

Output: Data/lux_source_features.csv (column-identical to uk_ev_features_full.csv).
Validation: non-weather features checked strictly vs frozen scalers; weather
deviation reported (not fatal).
"""
import sys, json
import numpy as np
import pandas as pd
import requests
import joblib

sys.path.insert(0, ".")
from coldstart_transfer.features import engineer_features, SEQ_FEATURES, WEATHER_VARS

LAT, LON = 49.70673, 6.48714
HOURLY = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
          "cloud_cover", "temperature_2m", "wind_speed_10m"]
EXPECT_TRAIN_ROWS = 38883
SPLIT_TS = pd.Timestamp("2025-10-20 22:45:00+00:00")  # max_ts - 90d (validated offline)


def load_ev_source():
    df = pd.read_csv("Data/ECC_master_PV_EMOB1_EMOB2_15min.csv")
    df["timestamp"] = pd.to_datetime(df["Started at"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    act = df[df["EV_TotalConsumption_kW"].notna()].copy()  # EMOB2-active window
    act = act.rename(columns={"EV_TotalConsumption_kW": "ev_kw"})
    return act[["timestamp", "ev_kw"]].reset_index(drop=True)


def fetch_weather(start, end):
    print(f"[wx] Open-Meteo archive {start.date()}..{end.date()} @({LAT},{LON})")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "start_date": str(start.date()), "end_date": str(end.date()),
        "hourly": ",".join(HOURLY), "timezone": "UTC",
    }, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    wx = pd.DataFrame({"timestamp": pd.to_datetime(h["time"], utc=True),
                       **{v: h[v] for v in HOURLY}})
    print(f"[wx] {len(wx)} hourly rows")
    return wx


def main():
    ev = load_ev_source()
    print(f"[ev] {len(ev)} rows  [{ev.timestamp.iloc[0]} .. {ev.timestamp.iloc[-1]}]  "
          f"ev_kw max={ev.ev_kw.max()}")
    wx = fetch_weather(ev.timestamp.iloc[0] - pd.Timedelta(days=1),
                       ev.timestamp.iloc[-1] + pd.Timedelta(days=1))
    # hourly -> 15-min on the EV grid via time interpolation
    wx = wx.set_index("timestamp").sort_index()
    grid = pd.DatetimeIndex(ev["timestamp"])
    wx15 = wx.reindex(wx.index.union(grid)).interpolate("time").reindex(grid)
    wx15 = wx15.ffill().bfill()

    raw = ev.copy()
    for v in HOURLY:
        raw[v] = wx15[v].to_numpy()
    feat = engineer_features(raw)   # builds the 21 SEQ_FEATURES

    # --- validation ---
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sc = scalers["scaler_seq"]
    dmin, dmax = np.asarray(sc.data_min_, float), np.asarray(sc.data_max_, float)
    ts = pd.to_datetime(feat["timestamp"], utc=True)
    train = feat[(ts < SPLIT_TS).to_numpy()]
    n_train = len(train)
    print(f"[validate] train_rows={n_train} expected={EXPECT_TRAIN_ROWS} "
          f"MATCH={n_train == EXPECT_TRAIN_ROWS}")
    print("[validate] per-feature min/max vs frozen scaler:")
    for i, c in enumerate(SEQ_FEATURES):
        col = train[c].to_numpy(float)
        tag = "WX~" if c in WEATHER_VARS else "   "
        flag = "" if abs(col.min()-dmin[i]) < 1e-2 and abs(col.max()-dmax[i]) < 1e-2 else "  <-- differs"
        print(f"  {tag}{c:22s} recon[{col.min():.3f},{col.max():.3f}] "
              f"scaler[{dmin[i]:.3f},{dmax[i]:.3f}]{flag}")

    feat.to_csv("Data/lux_source_features.csv", index=False)
    json.dump({"n_rows": len(feat), "n_train": int(n_train), "split_ts": str(SPLIT_TS),
               "ev_kw_max": float(ev.ev_kw.max()), "weather": "open-meteo re-fetch (approx)"},
              open("Data/lux_source_features_provenance.json", "w"), indent=2)
    print(f"[out] wrote Data/lux_source_features.csv ({len(feat)} rows)")


if __name__ == "__main__":
    main()
