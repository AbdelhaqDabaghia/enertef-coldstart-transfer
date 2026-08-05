"""
Multi-country #3, controlled: does a RESIDUAL distance->gap effect survive once we
control for the target's INTRINSIC predictability? Control = a model-free
climatology baseline (mean per 10-day doy bin x 15-min tod, fit train-only),
measured as RMSE in per-unit space (no mean-normalisation -> no low-solar blowup).
Then partial Spearman/Pearson of distance vs warm-start error | climatology error.
Outliers EE (2 MW, unreliable) and NL (near-zero solar, pathological) dropped.
"""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

def clim_rmse_pu(df):
    ts = pd.to_datetime(df.timestamp, utc=True)
    key = (ts.dt.dayofyear // 10).astype(str) + "_" + (ts.dt.hour*4 + ts.dt.minute//15).astype(str)
    ho = np.arange(len(df) - 14*96, len(df)); tr = np.arange(0, len(df) - 14*96)
    m = df.iloc[tr].assign(k=key.iloc[tr]).groupby("k")["pv_kw"].mean()
    fb = float(df.iloc[tr]["pv_kw"].mean())
    pred = key.iloc[ho].map(m).fillna(fb).to_numpy()
    true = df.iloc[ho]["pv_kw"].to_numpy()
    return float(np.sqrt(np.mean((pred - true)**2))), float(true.mean())

mc = pd.read_csv("Data/results/multicountry.csv").set_index("country")
rows = []
for path in sorted(glob.glob("Data/multicountry/*_features.csv")):
    cc = os.path.basename(path).replace("_features.csv", "")
    if cc not in mc.index: continue
    clim, mean_ho = clim_rmse_pu(pd.read_csv(path))
    b3_pu = mc.loc[cc, "b3_nrmse"] * mean_ho          # RMSE in per-unit
    rows.append(dict(country=cc, weather_mmd=mc.loc[cc, "weather_mmd"],
                     pv_shape_W=mc.loc[cc, "pv_shape_W"], clim_rmse=round(clim, 4),
                     b3_rmse_pu=round(b3_pu, 4)))
R = pd.DataFrame(rows)
R = R[~R.country.isin(["EE", "NL"])].reset_index(drop=True)   # drop unreliable/pathological
print(R.sort_values("weather_mmd").to_string(index=False))

def partial(x, y, z):
    # residualise x and y on z (linear), correlate residuals
    Z = np.c_[np.ones(len(z)), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)

for xc in ["weather_mmd", "pv_shape_W"]:
    r0, p0 = spearmanr(R[xc], R.b3_rmse_pu)
    pr, pp = partial(R[xc].to_numpy(float), R.b3_rmse_pu.to_numpy(float), R.clim_rmse.to_numpy(float))
    print(f"{xc}: raw Spearman={r0:+.3f} (p={p0:.3f}) | PARTIAL Pearson |clim = {pr:+.3f} (p={pp:.3f}), n={len(R)}")
# sanity: does climatology explain the error? (confound strength)
cr, cp = pearsonr(R.clim_rmse, R.b3_rmse_pu)
print(f"[confound] corr(climatology, warm-start error) = {cr:+.3f} (p={cp:.3f})")
