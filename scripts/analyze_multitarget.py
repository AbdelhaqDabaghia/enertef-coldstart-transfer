"""Multi-target analysis (#7): relate source->target domain distance to transfer
performance across the PV target sites. Merges pv_multitarget.csv (performance)
with pv_multitarget_distance.csv (distance), prints a table + Spearman rho, and
writes fig_multitarget.png."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

perf = pd.read_csv("Data/results/pv_multitarget.csv")
perf["nrmse_cap"] = perf["target_rmse_kw"] / perf["target_mean_kw"]   # capacity stored in target_mean_kw
dist = pd.read_csv("Data/results/pv_multitarget_distance.csv")

b3 = perf[perf.baseline == "B3"].groupby("notes")["nrmse_cap"].agg(["mean", "std"]).rename_axis("target").reset_index()
b0 = perf[perf.baseline == "B0"].groupby("notes")["nrmse_cap"].agg(m0="mean").reset_index().rename(columns={"notes": "target"})
m = dist.merge(b3, on="target").merge(b0, on="target")
m["improvement"] = m["m0"] / m["mean"]     # x-fold benefit of warm-start
print(m[["target", "capacity_kw", "scale_ratio", "W_pvkw_shape", "weather_mmd2", "mean", "std", "improvement"]].to_string(index=False))

for xcol in ["scale_ratio", "W_pvkw_shape"]:
    rho, p = spearmanr(m[xcol], m["mean"])
    print(f"Spearman( {xcol} , warm-start nRMSE_cap ) = {rho:+.3f} (p={p:.3f})")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.errorbar(m["scale_ratio"], m["mean"], yerr=m["std"], fmt="o", ms=8,
            color="#0072B2", capsize=3, lw=1.5)
for _, r in m.iterrows():
    ax.annotate(r["target"], (r["scale_ratio"], r["mean"]), xytext=(6, 3),
                textcoords="offset points", fontsize=8, color="#444")
ax.set_xlabel("source$\\rightarrow$target scale ratio (LU/target PV capacity)")
ax.set_ylabel("warm-start transfer error  (nRMSE by capacity)")
ax.set_title("Multi-target: domain distance vs. transfer quality (PV, N=90)")
ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("Data/results/figures/fig_multitarget.png", dpi=130); fig.savefig("Data/results/figures/fig_multitarget.pdf"); plt.close(fig)
print("wrote Data/results/figures/fig_multitarget.png")
