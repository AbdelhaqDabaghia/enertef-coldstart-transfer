"""Per-target domain distance (LU source PV -> each Konstanz PV site) for the
multi-target experiment. Writes Data/results/pv_multitarget_distance.csv."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

W = ["shortwave_radiation","direct_radiation","diffuse_radiation","cloud_cover","temperature_2m","wind_speed_10m"]
src = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
src_cap = src.pv_kw.max()

def wn(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.std(np.concatenate([a, b])) or 1.0
    return wasserstein_distance(a / s, b / s)

def mmd2(X, Y, n=2000, seed=0):
    r = np.random.default_rng(seed)
    X = X[r.choice(len(X), min(n, len(X)), False)]; Y = Y[r.choice(len(Y), min(n, len(Y)), False)]
    Z = np.vstack([X, Y]); d2 = np.sum((Z[:400, None]-Z[None, :400])**2, -1); g = 1/(np.median(d2[d2>0])+1e-9)
    k = lambda A, B: np.exp(-g*np.sum((A[:, None]-B[None])**2, -1))
    return float(k(X, X).mean()+k(Y, Y).mean()-2*k(X, Y).mean())

rows = []
for path in sorted(glob.glob("Data/pv_target/konstanz_*features.csv")):
    name = os.path.basename(path).replace("konstanz_", "").replace("_features.csv", "")
    t = pd.read_csv(path); tc = t.pv_kw.max()
    # pv_kw distances: raw (scale-normalized) and SHAPE (each normalized by own capacity)
    w_raw = wn(src.pv_kw, t.pv_kw)
    w_shape = wasserstein_distance((src.pv_kw/src_cap).to_numpy(), (t.pv_kw/tc).to_numpy())
    # weather MMD (standardized on pooled)
    Wm = np.vstack([src[W].to_numpy(float), t[W].to_numpy(float)]); mu, sd = Wm.mean(0), Wm.std(0)+1e-9
    wmmd = mmd2((src[W].to_numpy(float)-mu)/sd, (t[W].to_numpy(float)-mu)/sd)
    rows.append(dict(target=name, capacity_kw=round(tc, 2), scale_ratio=round(src_cap/tc, 1),
                     W_pvkw=round(w_raw, 4), W_pvkw_shape=round(w_shape, 5), weather_mmd2=round(wmmd, 4)))
    print(rows[-1])

pd.DataFrame(rows).to_csv("Data/results/pv_multitarget_distance.csv", index=False)
print("wrote Data/results/pv_multitarget_distance.csv")
