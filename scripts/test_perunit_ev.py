"""
Test #1 (per-unit normalisation) + #2 (Fisher comparability) on EV transfer.

Hypothesis: the 6x demand-scale gap (and the 211x source/target Fisher mismatch)
is an artefact of scaling power in kW. In the power-engineering PER-UNIT system,
each site's power features are divided by its rated capacity -> dimensionless,
cross-site comparable.

We (1) train an EV source model FROM SCRATCH in per-unit on the LU source, then
(2) transfer to UK in per-unit: B0 (scratch) vs B3 (warm from the per-unit source),
and (3) measure the source/target Fisher ratio in per-unit.
Reported in kW (pred x target base) for comparability with the min-max pipeline
(min-max B3 @N=30 was ~12 kW RMSE / 0.525 nRMSE).
"""
import time, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.features import SEQ_FEATURES, NEXT_EXO
from coldstart_transfer.model import build_ev_model
from coldstart_transfer import ewc as ewc_mod

LB = 672
POWER = ["ev_kw", "lag_1", "lag_4", "lag_96", "lag_672", "roll_1h_mean", "roll_6h_mean", "roll_24h_mean"]
WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover", "temperature_2m", "wind_speed_10m"]

def per_unit_frame(df, base, wx_min, wx_rng):
    d = pd.DataFrame()
    for c in SEQ_FEATURES:
        if c in POWER:
            d[c] = df[c].to_numpy(float) / base
        elif c in WEATHER:
            i = WEATHER.index(c); d[c] = (df[c].to_numpy(float) - wx_min[i]) / wx_rng[i]
        else:
            d[c] = df[c].to_numpy(float)  # time features already in [-1,1]
    return d

def windows(d, y_src):
    seq = d[SEQ_FEATURES].to_numpy(np.float32); nxt = d[NEXT_EXO].to_numpy(np.float32)
    n = len(d) - LB
    Xs = np.zeros((n, LB, len(SEQ_FEATURES)), np.float32); Xn = np.zeros((n, len(NEXT_EXO)), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        Xs[i] = seq[i:i+LB]; Xn[i] = nxt[i+LB]; y[i] = y_src[i+LB]
    return Xs, Xn, y

src = pd.read_csv("Data/lux_source_features.csv")
tgt = pd.read_csv("Data/uk_ev_features_full.csv")
base_src = float(np.quantile(src.ev_kw, 0.999)); base_tgt = float(np.quantile(tgt.ev_kw, 0.999))
wx_min = src[WEATHER].min().to_numpy(float); wx_rng = (src[WEATHER].max() - src[WEATHER].min()).to_numpy(float) + 1e-9
print(f"[per-unit] base_src={base_src:.1f}kW base_tgt={base_tgt:.1f}kW (scale ratio {base_src/base_tgt:.1f}x collapses to 1.0 in per-unit)")

sd = per_unit_frame(src, base_src, wx_min, wx_rng); sX, sN, sy = windows(sd, sd.ev_kw.to_numpy(np.float32))
td = per_unit_frame(tgt, base_tgt, wx_min, wx_rng); tX, tN, ty = windows(td, td.ev_kw.to_numpy(np.float32))
# splits: source train (all but last 90d) ; target first-30d train + last-14d holdout
s_tr = slice(0, len(sy) - 90*96)
t_ho = slice(len(ty) - 14*96, len(ty)); t_tr = slice(0, 30*96)

# 1) train per-unit SOURCE model from scratch
keras.utils.set_random_seed(0)
src_model = build_ev_model(seed=0)
src_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
t0 = time.time()
src_model.fit([sX[s_tr], sN[s_tr]], sy[s_tr], epochs=12, batch_size=128, verbose=0)
print(f"[per-unit] source model trained ({time.time()-t0:.0f}s)")

# 2) Fisher ratio in per-unit (source vs target N=30)
Fs = ewc_mod.estimate_fisher(src_model, sX[s_tr], sN[s_tr], m_points=500, batch_size=64)
Ft = ewc_mod.estimate_fisher(src_model, tX[t_tr], tN[t_tr], m_points=500, batch_size=64)
rs = sum(float(np.sum(f.numpy())) for f in Fs); rt = sum(float(np.sum(f.numpy())) for f in Ft)
print(f"[per-unit] Fisher ratio source/target = {rs/rt:.2f}x   (min-max pipeline was 211x)")

# 3) transfer: B0 scratch vs B3 warm, target per-unit N=30, 2 seeds
def rmse_kw(m):
    p = np.clip(m.predict([tX[t_ho], tN[t_ho]], verbose=0).reshape(-1), 0, None)
    return float(np.sqrt(np.mean(((p - ty[t_ho]) * base_tgt) ** 2)))
rows = []
mean_kw = float(np.mean(ty[t_ho]) * base_tgt)
for seed in range(5):
    for cond in ("B0", "B3"):
        keras.utils.set_random_seed(seed)
        m = build_ev_model(seed=seed)
        if cond == "B3":
            m.set_weights(src_model.get_weights())
        m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
        m.fit([tX[t_tr], tN[t_tr]], ty[t_tr], epochs=10, batch_size=64, verbose=0)
        rk = rmse_kw(m); rows.append((cond, seed, rk, rk/mean_kw))
        print(f"[per-unit {cond} s={seed}] target RMSE={rk:.2f}kW  nRMSE={rk/mean_kw:.3f}  (min-max B3~12kW/0.525)", flush=True)
import csv
with open("Data/results/test_perunit_ev.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cond", "seed", "rmse_kw", "nrmse"]); w.writerows(rows)
for c in ("B0", "B3"):
    v = [r[3] for r in rows if r[0] == c]
    print(f"{c}: nRMSE {np.mean(v):.3f}+-{np.std(v):.3f}  (min-max B3=0.525)")
print(f"Fisher ratio (per-unit) = {rs/rt:.2f}x  (min-max 211x)")
print("DONE")
