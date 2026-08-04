"""
build_pv_source.py -- reconstruct the Luxembourg SOURCE PV features for the PV
replication (B1/B5/retention), mirroring build_lux_source.py.

PV target = ECC_master PV_TotalProduction_kW (max 751.2 == frozen scaler exactly);
weather re-fetched from Open-Meteo for Copal Dudelange. Output validated against
the frozen PV scalers. Writes Data/pv_target/lux_pv_source_features.csv.
"""
import sys, numpy as np, pandas as pd, requests, joblib
sys.path.insert(0, ".")
from coldstart_transfer.pv import engineer_pv_features, PV_SEQ_FEATURES, PV_WEATHER

LAT, LON = 49.70673, 6.48714           # Copal Dudelange
EXPECT_TRAIN = 71043
SPLIT_TS = pd.Timestamp("2025-10-20 22:45:00+00:00")   # max_ts - 90d


def main():
    df = pd.read_csv("Data/ECC_master_PV_EMOB1_EMOB2_15min.csv")
    df["timestamp"] = pd.to_datetime(df["Started at"], utc=True)
    pv = df[df["PV_TotalProduction_kW"].notna()].sort_values("timestamp")
    pv = pv.rename(columns={"PV_TotalProduction_kW": "pv_kw"})[["timestamp", "pv_kw"]].reset_index(drop=True)
    print(f"[pv-src] {len(pv)} rows [{pv.timestamp.iloc[0]} .. {pv.timestamp.iloc[-1]}] max={pv.pv_kw.max()}")

    hourly = PV_WEATHER
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "start_date": str((pv.timestamp.iloc[0] - pd.Timedelta(days=1)).date()),
        "end_date": str((pv.timestamp.iloc[-1] + pd.Timedelta(days=1)).date()),
        "hourly": ",".join(hourly), "timezone": "UTC"}, timeout=180)
    r.raise_for_status()
    h = r.json()["hourly"]
    wx = pd.DataFrame({"timestamp": pd.to_datetime(h["time"], utc=True), **{v: h[v] for v in hourly}})
    wx = wx.set_index("timestamp").sort_index()
    grid = pd.DatetimeIndex(pv["timestamp"])
    wx15 = wx.reindex(wx.index.union(grid)).interpolate("time").reindex(grid).ffill().bfill()
    for c in PV_WEATHER:
        pv[c] = wx15[c].to_numpy()

    feat = engineer_pv_features(pv)
    # validate vs frozen PV scalers
    sc = joblib.load("Data/models/pv_scalers.joblib")["scaler_seq"]
    ts = pd.to_datetime(feat["timestamp"], utc=True)
    train = feat[(ts < SPLIT_TS).to_numpy()]
    print(f"[validate] train_rows={len(train)} expected={EXPECT_TRAIN} MATCH={len(train)==EXPECT_TRAIN}")
    ok = True
    for i, c in enumerate(PV_SEQ_FEATURES):
        col = train[c].to_numpy(float)
        if abs(col.min()-sc.data_min_[i]) > 1e-2 or abs(col.max()-sc.data_max_[i]) > 1e-2:
            print(f"  {c:12s} recon[{col.min():.3f},{col.max():.3f}] scaler[{sc.data_min_[i]:.3f},{sc.data_max_[i]:.3f}] <-- differs")
            ok = False
    print("[validate] all-feature min/max match:", ok)
    feat.to_csv("Data/pv_target/lux_pv_source_features.csv", index=False)
    print(f"[out] wrote Data/pv_target/lux_pv_source_features.csv ({len(feat)} rows)")


if __name__ == "__main__":
    main()
