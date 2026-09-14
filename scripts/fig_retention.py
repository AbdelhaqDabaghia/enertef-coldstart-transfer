"""
fig_retention.py -- retained-task error under both feature pipelines.

Two panels on a shared vertical scale. Each line joins one seed's raw error to
its gauge-corrected error, so the reader sees the correction act on individual
runs rather than on a mean.

Left, the leaky pipeline: every condition improves under correction and the
ordering of the first two reverses -- the result the manuscript previously led
with. Right, the causal pipeline: the correction moves almost nothing except
for rehearsal, the two anchoring conditions collapse onto each other, and only
rehearsal falls below naive persistence.

Reads both e13 CSVs. Writes paper/v4/figures/fig_retention.{pdf,svg,png}.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle as S

LEAKY = os.path.join("Data", "results", "e13_gauge_decomposition.csv")
CAUSAL = os.path.join("Data", "results", "corrected_causal_pipeline",
                      "e13_gauge_decomposition.csv")
ORDER = ["B3_warm", "MAS", "B5_replay"]
PERSISTENCE = 0.9972          # same holdout, uses no features


def panel(ax, df, title, show_ylabel):
    S.grid(ax)
    d = df[df.seed >= 0]

    for i, c in enumerate(ORDER):
        k = S.KEY[c]
        sub = d[d.condition == c]
        x0, x1 = i - 0.17, i + 0.17

        # one line per seed: raw -> gauge-corrected
        for _, r in sub.iterrows():
            ax.plot([x0, x1], [r["raw"], r["gauged"]],
                    color=S.EDGE[k], linewidth=1.1, alpha=0.45, zorder=2)
        ax.scatter([x0] * len(sub), sub["raw"], s=70, color=S.FILL[k],
                   edgecolors=S.EDGE[k], linewidths=1.3, zorder=3)
        ax.scatter([x1] * len(sub), sub["gauged"], s=70, color="white",
                   edgecolors=S.EDGE[k], linewidths=1.8, zorder=3)

        # means, as short heavy bars
        ax.plot([x0 - 0.09, x0 + 0.09], [sub["raw"].mean()] * 2,
                color=S.EDGE[k], linewidth=3.0, zorder=4, solid_capstyle="butt")
        ax.plot([x1 - 0.09, x1 + 0.09], [sub["gauged"].mean()] * 2,
                color=S.EDGE[k], linewidth=3.0, zorder=4, solid_capstyle="butt")

    ax.axhline(PERSISTENCE, color=S.NEUTRAL, linewidth=2.0, linestyle="--",
               zorder=1)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(["Fine-\ntune", "Output-\nsensitivity",
                        "Bounded\nrehearsal"])
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.set_title(title, pad=14)
    if show_ylabel:
        ax.set_ylabel("Retained-task error (nRMSE)")
    else:
        ax.tick_params(labelleft=False)


def main():
    S.apply()
    L = pd.read_csv(LEAKY)
    C = pd.read_csv(CAUSAL)

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    panel(axes[0], L, "Leaky features", True)
    panel(axes[1], C, "Causal features", False)

    lo = min(L.gauged.min(), C.gauged.min())
    hi = max(L.raw.max(), C.raw.max())
    axes[0].set_ylim(lo - 0.07 * (hi - lo), hi + 0.10 * (hi - lo))

    # legend: what the two marker styles mean, plus the persistence line
    h = [plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                    markerfacecolor="#b9b9b9", markeredgecolor="#4e4e4e",
                    markeredgewidth=1.3, label="raw"),
         plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                    markerfacecolor="white", markeredgecolor="#4e4e4e",
                    markeredgewidth=1.8, label="gauge-corrected"),
         plt.Line2D([], [], color=S.NEUTRAL, linewidth=2.0, linestyle="--",
                    label="naive persistence")]
    axes[0].legend(handles=h, loc="upper center",
                   bbox_to_anchor=(1.03, 1.20), ncol=3, frameon=False)

    S.save(fig, "fig_retention")


if __name__ == "__main__":
    main()
