"""
Test #6 (clear-sky index) for PV transfer -- CONSOLIDATED: train-only envelope
(no holdout leak) and 5 transfer seeds.

Forecast csi = power / clear-sky-envelope, envelope = 95th-pct power per
(10-day doy bin, 15-min tod) fit on TRAINING data only, applied to all rows
(fallback = global training median). Report RMSE in kW (pred_csi x envelope).
Raw-power B3 @N=30 was ~6.85 kW.
"""
import time, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.pv import build_pv_model, PV_SEQ_FEATURES, PV_NEXT_EXO, PV_LOOKBACK as LB, PV_WEATHER

def env_key(df):
    ts = pd.to_datetime(df.timestamp, utc=True)
    return (ts.dt.dayofyear // 10).astype(int).astype(str) + "_" + (ts.dt.hour*4 + ts.dt.minute//15).astype(str)

def fit_envelope(df, fit_idx):
    """Fit the clear-sky quantile map on df.iloc[fit_idx]; return envelope for ALL rows."""
    k = env_key(df)
    sub = df.iloc[fit_idx].assign(k=k.iloc[fit_idx])
    q = sub.groupby("k")["pv_kw"].quantile(0.95).clip(lower=0.05)
    fallback = max(float(sub["pv_kw"].median()), 0.05)
    env = k.map(q).fillna(fallback).to_numpy(float)
    return np.maximum(env, 0.05)

def to_csi(df, fit_idx):
    env = fit_envelope(df, fit_idx)
    d = df.copy(); d["pv_kw"] = np.clip(df.pv_kw.to_numpy(float) / env, 0, 1.3)
    return d, env

def windows(d, yv):
    seq = d[PV_SEQ_FEATURES].to_numpy(np.float32); nxt = d[PV_NEXT_EXO].to_numpy(np.float32)
    n = len(d) - LB
    Xs = np.zeros((n, LB, len(PV_SEQ_FEATURES)), np.float32); Xn = np.zeros((n, len(PV_NEXT_EXO)), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        Xs[i] = seq[i:i+LB]; Xn[i] = nxt[i+LB]; y[i] = yv[i+LB]
    return Xs, Xn, y

src = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
tgt = pd.read_csv("Data/pv_target/konstanz_pv_features.csv")
# train-only envelope: source fit on all-but-last-90d; target fit on all-but-last-14d
s_fit = np.arange(0, len(src) - 90*96); t_fit = np.arange(0, len(tgt) - 14*96)
sd, s_env = to_csi(src, s_fit); td, t_env = to_csi(tgt, t_fit)
wmin = sd[PV_WEATHER].min().to_numpy(float); wrng = (sd[PV_WEATHER].max()-sd[PV_WEATHER].min()).to_numpy(float)+1e-9
for d in (sd, td):
    for i, c in enumerate(PV_WEATHER):
        d[c] = (d[c].to_numpy(float)-wmin[i])/wrng[i]

sX, sN, sy = windows(sd, sd.pv_kw.to_numpy(np.float32))
tX, tN, ty = windows(td, td.pv_kw.to_numpy(np.float32)); t_env_w = t_env[LB:]
s_tr = slice(0, len(sy)-90*96); t_ho = slice(len(ty)-14*96, len(ty)); t_tr = slice(0, 30*96)

keras.utils.set_random_seed(0)
srcm = build_pv_model(seed=0); srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
t0=time.time(); srcm.fit([sX[s_tr], sN[s_tr]], sy[s_tr], epochs=12, batch_size=128, verbose=0)
print(f"[csi] source PV model trained (train-only envelope) ({time.time()-t0:.0f}s)")

def rmse_kw(m):
    p = np.clip(m.predict([tX[t_ho], tN[t_ho]], verbose=0).reshape(-1), 0, 1.3)
    return float(np.sqrt(np.mean(((p - ty[t_ho]) * t_env_w[t_ho])**2)))
rows=[]
for seed in range(5):
    for cond in ("B0","B3"):
        keras.utils.set_random_seed(seed)
        m = build_pv_model(seed=seed)
        if cond=="B3": m.set_weights(srcm.get_weights())
        m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
        m.fit([tX[t_tr], tN[t_tr]], ty[t_tr], epochs=10, batch_size=64, verbose=0)
        rk=rmse_kw(m); rows.append((cond,seed,rk)); print(f"[csi {cond} s={seed}] RMSE={rk:.3f}kW", flush=True)
import csv
with open("Data/results/test_clearsky_pv.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["cond","seed","rmse_kw"]); w.writerows(rows)
for c in ("B0","B3"):
    v=[r[2] for r in rows if r[0]==c]; print(f"{c}: {np.mean(v):.3f}+-{np.std(v):.3f} kW  (raw B3 ~6.85)")
print("DONE")
