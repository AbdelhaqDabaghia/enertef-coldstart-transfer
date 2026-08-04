"""
domain_distance.py -- characterise the SOURCE (Luxembourg) vs TARGET (UK) domain
shift for RQ3, so the B1-vs-B4 gap can be related to how far the target is.

Metrics (single target => characterise/report, not vary):
  - 1-D Wasserstein per variable: ev_kw + each weather var (raw + std-normalized)
  - RBF-MMD on the standardized 6-D weather vector (median-heuristic bandwidth)
Also basic mean/std for context.
"""
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "cloud_cover", "temperature_2m", "wind_speed_10m"]

src = pd.read_csv("Data/lux_source_features.csv")
tgt = pd.read_csv("Data/uk_ev_features_full.csv")


def wd_norm(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.std(np.concatenate([a, b])) or 1.0
    return wasserstein_distance(a, b), wasserstein_distance(a / s, b / s)


print("=== marginal Wasserstein (source LU vs target UK) ===")
print(f"{'variable':22s} {'W_raw':>10s} {'W_norm':>8s}   src(mean±std)        tgt(mean±std)")
for c in ["ev_kw"] + WEATHER:
    a, b = src[c].to_numpy(float), tgt[c].to_numpy(float)
    wr, wn = wd_norm(a, b)
    print(f"{c:22s} {wr:10.3f} {wn:8.3f}   "
          f"{a.mean():7.2f}±{a.std():6.2f}   {b.mean():7.2f}±{b.std():6.2f}")


def rbf_mmd2(X, Y, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    Z = np.vstack([X, Y])
    # median heuristic bandwidth on pooled pairwise distances (subsample)
    d2 = np.sum((Z[:500, None, :] - Z[None, :500, :]) ** 2, -1)
    gamma = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
    def k(A, B):
        return np.exp(-gamma * np.sum((A[:, None, :] - B[None, :, :]) ** 2, -1))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


# standardize weather by pooled stats, then MMD
W = np.vstack([src[WEATHER].to_numpy(float), tgt[WEATHER].to_numpy(float)])
mu, sd = W.mean(0), W.std(0) + 1e-9
Xs = (src[WEATHER].to_numpy(float) - mu) / sd
Ys = (tgt[WEATHER].to_numpy(float) - mu) / sd
mmd = rbf_mmd2(Xs, Ys)
print(f"\n=== RBF-MMD^2 on standardized 6-D weather = {mmd:.4f} "
      f"(0=identical; larger=more shifted) ===")
print(f"ev_kw scale ratio (LU/UK max) = {src.ev_kw.max()/max(tgt.ev_kw.max(),1e-9):.1f}x")
