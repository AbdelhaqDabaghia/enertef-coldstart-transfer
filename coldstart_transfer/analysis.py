"""
analysis.py -- generate the three cold-start figures from the logged results.

  (a) RQ1  nRMSE-vs-N per baseline (B0/B3/B4)   -> fig_rq1_coldstart.png
  (b) RQ2  stability-plasticity tradeoff        -> fig_rq2_tradeoff.png
  (c) RQ3  B1-vs-B4 + domain distance           -> fig_rq3_fisher.png

Design: Okabe-Ito CVD-safe categorical palette, one hue per condition in a FIXED
order (never cycled), single axis per chart, std error bars over seeds, legend +
selective direct labels, recessive grid.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "Data/results/figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (colorblind-safe); fixed entity->hue mapping, consistent across figs.
C = {"B0": "#D55E00", "B3": "#0072B2", "B4": "#56B4E9",
     "B1": "#E69F00", "B5": "#009E73"}
LBL = {"B0": "B0 from-scratch", "B3": "B3 warm, no-EWC",
       "B1": "B1 warm+EWC (src Fisher)", "B4": "B4 warm+EWC (tgt Fisher)",
       "B5": "B5 warm+replay"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True})


def fig_rq1():
    d = pd.read_csv("Data/results/rq1_coldstart.csv")
    g = d.groupby(["baseline", "n_days"])["target_nrmse"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in ["B0", "B3", "B4"]:
        s = g[g.baseline == b].sort_values("n_days")
        ax.errorbar(s.n_days, s["mean"], yerr=s["std"], marker="o", ms=6, lw=2,
                    color=C[b], capsize=3, label=LBL[b])
        ax.annotate(b, (s.n_days.values[-1], s["mean"].values[-1]),
                    xytext=(6, 0), textcoords="offset points", color=C[b],
                    va="center", fontweight="bold")
    ax.set_xscale("log"); ax.set_xticks([1, 7, 30, 60, 90])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("target training data available  (N days, log scale)")
    ax.set_ylabel("target nRMSE  (lower = better)")
    ax.set_title("RQ1 — cold-start transfer benefit (Luxembourg → UK)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rq1_coldstart.png"); plt.close(fig)
    print("wrote fig_rq1_coldstart.png")


def fig_rq2():
    d = pd.read_csv("Data/results/rq2_rq3_source.csv")
    d["source_retention_nrmse"] = pd.to_numeric(d["source_retention_nrmse"], errors="coerce")
    g = d.groupby("baseline").agg(
        tx=("target_nrmse", "mean"), txs=("target_nrmse", "std"),
        ry=("source_retention_nrmse", "mean"), rys=("source_retention_nrmse", "std"))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    # per-condition label offsets to avoid the B3/B4 collision (they nearly overlap)
    off = {"B0": (8, 4), "B1": (8, 4), "B5": (8, 4),
           "B3": (10, -16), "B4": (10, 10)}
    for b in ["B0", "B3", "B1", "B4", "B5"]:
        r = g.loc[b]
        ax.errorbar(r.ry, r.tx, xerr=r.rys, yerr=r.txs, marker="o", ms=9,
                    color=C[b], capsize=3, lw=1.5)
        ax.annotate(LBL[b], (r.ry, r.tx), xytext=off[b], textcoords="offset points",
                    color=C[b], fontsize=9)
    ax.set_xlabel("source-retention nRMSE  (stability, lower = better)")
    ax.set_ylabel("target nRMSE  (plasticity, lower = better)")
    ax.set_title("RQ2 — stability–plasticity tradeoff  (N=30, 3 seeds)")
    ax.annotate("better", (0.02, 0.02), xycoords="axes fraction", fontsize=9,
                color="#666", ha="left", va="bottom",
                arrowprops=None)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rq2_tradeoff.png"); plt.close(fig)
    print("wrote fig_rq2_tradeoff.png")


def fig_rq3():
    d = pd.read_csv("Data/results/rq2_rq3_source.csv")
    d["source_retention_nrmse"] = pd.to_numeric(d["source_retention_nrmse"], errors="coerce")
    g = d.groupby("baseline").agg(
        tx=("target_nrmse", "mean"), txs=("target_nrmse", "std"),
        ry=("source_retention_nrmse", "mean"), rys=("source_retention_nrmse", "std"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    metrics = ["target nRMSE", "source-retention nRMSE"]
    x = np.arange(len(metrics)); w = 0.36
    for i, b in enumerate(["B1", "B4"]):
        r = g.loc[b]
        vals = [r.tx, r.ry]; errs = [r.txs, r.rys]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
               color=C[b], label=LBL[b])
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("nRMSE  (lower = better)")
    ax.set_title("RQ3 — Fisher transferability: source vs target (λ=100)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.97, "domain shift (LU→UK):\nev_kw W=0.62 (6× scale)\nwind W=0.63\n"
            "weather MMD²=0.034",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#cccccc"))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rq3_fisher.png"); plt.close(fig)
    print("wrote fig_rq3_fisher.png")


def fig_bench():
    """EWC-variants bench: every improved variant vs the B3/B5 baselines, on the
    same stability-plasticity axes. All variants land up-and-right of B3/B5 = worse."""
    d = pd.read_csv("Data/results/ewc_bench.csv")
    d["source_retention_nrmse"] = pd.to_numeric(d["source_retention_nrmse"], errors="coerce")
    g = d.groupby("notes").agg(
        tx=("target_nrmse", "mean"), txs=("target_nrmse", "std"),
        ry=("source_retention_nrmse", "mean"), rys=("source_retention_nrmse", "std"))
    # keep baselines + the best-lambda point of each variant family (min target)
    def best(prefix):
        sub = g[[i.startswith(prefix) for i in g.index]]
        return sub.loc[sub.tx.idxmin()], sub.tx.idxmin()
    points = {"B3 warm (baseline)": ("B3_warm", "#0072B2"),
              "B5 replay (baseline)": ("B5_replay", "#009E73"),
              "V1 empirical Fisher": (best("V1")[1], "#E69F00"),
              "V2 per-layer Fisher": (best("V2")[1], "#CC79A7"),
              "V3 EWC+replay hybrid": ("V3_hybrid", "#D55E00")}
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for lab, (key, col) in points.items():
        r = g.loc[key]
        ax.errorbar(r.ry, r.tx, xerr=r.rys, yerr=r.txs, marker="o", ms=9,
                    color=col, capsize=3, lw=1.5)
        ax.annotate(lab, (r.ry, r.tx), xytext=(8, 4), textcoords="offset points",
                    color=col, fontsize=9)
    ax.set_xlabel("source-retention nRMSE  (stability, lower = better)")
    ax.set_ylabel("target nRMSE  (plasticity, lower = better)")
    ax.set_title("EWC-variants bench — none beats warm-start or replay (N=30, 3 seeds)")
    ax.annotate("better", (0.02, 0.02), xycoords="axes fraction", color="#666",
                fontsize=9, ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_bench_ewc.png"); plt.close(fig)
    print("wrote fig_bench_ewc.png")


def fig_pv_rq1():
    d = pd.read_csv("Data/results/pv_rq1_coldstart.csv")
    g = d.groupby(["baseline", "n_days"])["target_rmse_kw"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in ["B0", "B3", "B4"]:
        s = g[g.baseline == b].sort_values("n_days")
        ax.errorbar(s.n_days, s["mean"], yerr=s["std"], marker="o", ms=6, lw=2,
                    color=C[b], capsize=3, label={"B0": "B0 from-scratch",
                    "B3": "B3 warm, no-EWC", "B4": "B4 warm+EWC"}[b])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 7, 30, 60, 90])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("target PV data available  (N days, log scale)")
    ax.set_ylabel("target RMSE (kW, log)  --  lower = better")
    ax.set_title("PV RQ1 --- cold-start transfer (Luxembourg $\\rightarrow$ Konstanz)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pv_rq1.png"); plt.close(fig)
    print("wrote fig_pv_rq1.png")


def fig_pv_rq2():
    d = pd.read_csv("Data/results/pv_rq2_rq3_source.csv")
    d["source_retention_nrmse"] = pd.to_numeric(d["source_retention_nrmse"], errors="coerce")
    g = d.groupby("baseline").agg(tx=("target_rmse_kw", "mean"), txs=("target_rmse_kw", "std"),
        ry=("source_retention_nrmse", "mean"), rys=("source_retention_nrmse", "std"))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    off = {"B0": (8, 4), "B1": (8, 6), "B5": (8, 4), "B3": (10, -14), "B4": (8, 8)}
    for b in ["B0", "B3", "B1", "B4", "B5"]:
        r = g.loc[b]
        ax.errorbar(r.ry, r.tx, xerr=r.rys, yerr=r.txs, marker="o", ms=9, color=C[b], capsize=3, lw=1.5)
        ax.annotate(LBL[b], (r.ry, r.tx), xytext=off[b], textcoords="offset points", color=C[b], fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("source-retention nRMSE  (stability, lower = better)")
    ax.set_ylabel("target RMSE (kW, log)  (plasticity, lower = better)")
    ax.set_title("PV RQ2 --- stability--plasticity trade-off (N=30, 3 seeds)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pv_rq2.png"); plt.close(fig)
    print("wrote fig_pv_rq2.png")


if __name__ == "__main__":
    fig_rq1(); fig_rq2(); fig_rq3(); fig_bench(); fig_pv_rq1(); fig_pv_rq2()
    print(f"[analysis] 6 figures -> {OUT}/")
