"""
fig_decoupling.py -- forecast accuracy against the decisions it drives.

The paper's central claim in one panel: each point is one (seed, day) pair,
positioned by the forecaster's MAE and by the realised cost of the schedule
built on it. Bounded rehearsal occupies the left of the plot on every pair --
it is unambiguously the more accurate forecaster -- yet its cloud sits at the
same height as the regulariser's. Moving left does not move you down.

Marginal box plots on each axis carry the summary so the reader does not have
to infer it from the cloud, and the annotation states the two test results.

Reads Data/results/corrected_causal_pipeline/e16_selection_oracle.csv.
Writes paper/v4/figures/fig_decoupling.{pdf,svg,png}.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import figstyle as S

SRC = os.path.join("Data", "results", "corrected_causal_pipeline",
                   "e16_selection_oracle.csv")
ORDER = ["B3_warm", "MAS", "B5_replay"]


def main():
    S.apply()
    d = pd.read_csv(SRC)
    mae = d.pivot_table(index=["seed", "day"], columns="condition", values="mae")
    obj = d.pivot_table(index=["seed", "day"], columns="condition",
                        values="objective")

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    S.grid(ax)

    for i, c in enumerate(ORDER):
        k = S.KEY[c]
        ax.scatter(mae[c], obj[c], s=90, color=S.FILL[k],
                   edgecolors=S.EDGE[k], linewidths=1.5, zorder=3,
                   label=S.LABEL[c], alpha=0.85)
        # condition mean, as a cross-hair
        ax.scatter([mae[c].mean()], [obj[c].mean()], s=420, marker="P",
                   color=S.EDGE[k], edgecolors="white", linewidths=2.0,
                   zorder=5)

    # marginal distributions
    for i, c in enumerate(ORDER):
        k = S.KEY[c]
        axx.boxplot(mae[c], orientation="horizontal", positions=[i], widths=0.62,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=S.FILL[k], color=S.EDGE[k],
                                  linewidth=1.4),
                    medianprops=dict(color=S.EDGE[k], linewidth=2.2),
                    whiskerprops=dict(color=S.EDGE[k], linewidth=1.4),
                    capprops=dict(color=S.EDGE[k], linewidth=1.4))
        axy.boxplot(obj[c], orientation="vertical", positions=[i], widths=0.62,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=S.FILL[k], color=S.EDGE[k],
                                  linewidth=1.4),
                    medianprops=dict(color=S.EDGE[k], linewidth=2.2),
                    whiskerprops=dict(color=S.EDGE[k], linewidth=1.4),
                    capprops=dict(color=S.EDGE[k], linewidth=1.4))

    # The marginals share axes with the main panel, so set_xticks([]) here
    # would strip the tick LOCATIONS from the scatter as well. Hide the
    # marginals' labels and tick marks only, and leave the locators alone.
    for a in (axx, axy):
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(which="both", length=0, labelbottom=False,
                      labelleft=False, labelright=False, labeltop=False)
    axx.set_ylim(-0.7, len(ORDER) - 0.3)
    axy.set_xlim(-0.7, len(ORDER) - 0.3)

    ax.set_xlabel("Forecast error, MAE (kW)")
    ax.set_ylabel("Realised control cost")

    xs = np.concatenate([mae[c].values for c in ORDER])
    ys = np.concatenate([obj[c].values for c in ORDER])
    S.limits(ax, xs, ys, pad=0.08)

    # the two numbers the figure exists to convey
    _, p_mae = wilcoxon(mae["B5_replay"], mae["MAS"])
    _, p_obj = wilcoxon(obj["B5_replay"], obj["MAS"])
    dm = (mae["MAS"] - mae["B5_replay"]).mean()
    n_all = int((mae["B5_replay"] < mae["MAS"]).sum())

    ax.annotate(
        "rehearsal is %.1f kW more accurate\n"
        "on %d/%d pairs  (p < 1e-5)\n"
        "yet control cost is unchanged\n"
        "(p = %.2f)" % (dm, n_all, len(mae), p_obj),
        xy=(0.975, 0.30), xycoords="axes fraction", va="top", ha="right",
        color="#4e4e4e",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                  edgecolor=S.NEUTRAL, linewidth=1.0, alpha=0.92))

    h, l = ax.get_legend_handles_labels()
    axx.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 2.15),
               ncol=3, frameon=False)

    S.save(fig, "fig_decoupling")


if __name__ == "__main__":
    main()
