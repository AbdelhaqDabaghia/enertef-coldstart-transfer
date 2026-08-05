"""Figure: multi-country distance -> transfer gap, coloured by intrinsic
predictability (climatology error) to show the mediation. Writes fig_multicountry.{png,pdf}."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def clim_and_mean(df):
    ts = pd.to_datetime(df.timestamp, utc=True)
    key = (ts.dt.dayofyear // 10).astype(str) + "_" + (ts.dt.hour*4 + ts.dt.minute//15).astype(str)
    ho = np.arange(len(df)-14*96, len(df)); tr = np.arange(0, len(df)-14*96)
    m = df.iloc[tr].assign(k=key.iloc[tr]).groupby("k")["pv_kw"].mean(); fb = float(df.iloc[tr]["pv_kw"].mean())
    pred = key.iloc[ho].map(m).fillna(fb).to_numpy(); true = df.iloc[ho]["pv_kw"].to_numpy()
    return float(np.sqrt(np.mean((pred-true)**2))), float(true.mean())

mc = pd.read_csv("Data/results/multicountry.csv").set_index("country")
rows = []
for path in sorted(glob.glob("Data/multicountry/*_features.csv")):
    cc = os.path.basename(path).replace("_features.csv", "")
    if cc not in mc.index or cc in ("EE", "NL"): continue
    clim, mean_ho = clim_and_mean(pd.read_csv(path))
    rows.append(dict(country=cc, dist=mc.loc[cc, "pv_shape_W"], err=mc.loc[cc, "b3_nrmse"]*mean_ho, clim=clim))
R = pd.DataFrame(rows)
rho, p = spearmanr(R.dist, R.err)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 4.4))
sc = ax.scatter(R.dist, R.err, c=R.clim, s=90, cmap="viridis", edgecolor="k", linewidth=0.5, zorder=3)
for _, r in R.iterrows():
    ax.annotate(r.country, (r.dist, r.err), xytext=(6, 3), textcoords="offset points", fontsize=9)
# trend line
b = np.polyfit(R.dist, R.err, 1); xs = np.linspace(R.dist.min(), R.dist.max(), 50)
ax.plot(xs, np.polyval(b, xs), "--", color="#888", lw=1.3, zorder=1)
ax.set_xlabel("source$\\rightarrow$target generation-shape distance ($W$)")
ax.set_ylabel("warm-start transfer error (RMSE, per-unit)")
ax.set_title(f"Multi-country transfer gap vs.\\ domain distance\n"
             f"Spearman $\\rho={rho:.2f}$ ($p={p:.3f}$, $n={len(R)}$) --- colour = intrinsic predictability")
cb = fig.colorbar(sc, ax=ax); cb.set_label("climatology RMSE (intrinsic difficulty)")
ax.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Data/results/figures/fig_multicountry.png"); fig.savefig("Data/results/figures/fig_multicountry.pdf")
print(f"wrote fig_multicountry (rho={rho:.3f} p={p:.3f} n={len(R)})")
