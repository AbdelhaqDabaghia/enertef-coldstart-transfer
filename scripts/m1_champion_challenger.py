"""
m1_champion_challenger.py -- does the best adaptation mechanism change across
cycles, and would selecting per cycle help?

DATA. enertef-cl-trainer/replay_results.csv: the 240 adaptation jobs of the
sequential study (4 strategies x 5 seeds x 12 cycles), with per-job
  new_mae        error on the recent window (plasticity)
  reference_mae  error on the fixed winter reference window (retention)
  verdict        PROMOTE / REGRESSION from the deployed 5% tolerance gate

METHODOLOGICAL LIMIT, stated up front. The four chains were each chained
independently: at cycle k the four models descend from four different histories.
A mixed trajectory that switches strategy at cycle k cannot be reconstructed
from these runs, because the cycle-(k+1) model would have had a different
parent. Everything below is therefore an ORACLE analysis -- an upper bound on
what per-cycle selection could achieve -- not a simulation of the deployed
system. Establishing the realised benefit requires re-running the stream with
the selection rule in the loop (stated as such in the manuscript).

What IS established without any assumption:
  (1) whether the winning strategy varies by cycle and seed. If one strategy
      always won, selection would be pointless and the contribution void.
  (2) how much a per-cycle oracle would gain over the best fixed strategy,
      bounding the value of selection.

Writes Data/results/m1_champion_challenger.csv and prints the analysis.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from itertools import combinations

SRC = os.environ.get("M1_SRC", "/mnt/c/dev/enertef-cl-trainer/replay_results.csv")
OUT = os.environ.get("OUT", "Data/results/m1_champion_challenger.csv")
STRATS = ["naive", "osr", "replay", "replay_osr"]
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0,0.5,1,2,1000").split(",")]
MID = list(range(3, 9))          # mid-stream cycles, as in the sequential study


def load():
    d = pd.read_csv(SRC)
    d = d[d.condition.isin(STRATS)].copy()
    # BWT(k) = ref_mae(k) - ref_mae(0), per strategy and seed
    base = (d[d.cycle == 0].set_index(["condition", "seed"]).reference_mae
            .rename("ref0"))
    d = d.join(base, on=["condition", "seed"])
    d["bwt"] = d.reference_mae - d.ref0
    return d


def winner_variability(d):
    """(1) Does the best strategy change across cycles and seeds?"""
    print("=" * 72)
    print("(1) WHO WINS, PER SEED AND CYCLE")
    print("=" * 72)
    for metric, label in [("reference_mae", "retention (reference MAE)"),
                          ("new_mae", "plasticity (recent MAE)")]:
        piv = d.pivot_table(index=["seed", "cycle"], columns="condition",
                            values=metric)
        piv = piv.dropna()
        win = piv.idxmin(axis=1)
        counts = win.value_counts()
        print(f"\n-- {label}: winner over {len(win)} (seed, cycle) pairs")
        for s in STRATS:
            n = int(counts.get(s, 0))
            print(f"     {s:12s} wins {n:3d}  ({100*n/len(win):5.1f} %)")
        # per-cycle winner, pooled over seeds
        pc = d.pivot_table(index="cycle", columns="condition", values=metric)
        print(f"     per-cycle winner (mean over seeds): "
              f"{list(pc.idxmin(axis=1).values)}")
    return


def oracle_selection(d):
    """(2) Upper bound: pick the best candidate at every (seed, cycle)."""
    print("\n" + "=" * 72)
    print("(2) ORACLE PER-CYCLE SELECTION vs FIXED STRATEGIES")
    print("=" * 72)
    rows = []
    fixed = {}
    for s in STRATS:
        sub = d[d.condition == s]
        fixed[s] = dict(
            ret_mid=sub[sub.cycle.isin(MID)].bwt.mean(),
            ret_all=sub.bwt.mean(),
            plast=sub[sub.cycle == sub.cycle.max()].new_mae.mean())
        rows.append(dict(strategy=s, alpha=np.nan, **fixed[s]))
        print(f"  fixed {s:12s} BWT_mid={fixed[s]['ret_mid']:+7.3f}  "
              f"BWT_all={fixed[s]['ret_all']:+7.3f}  "
              f"final plasticity={fixed[s]['plast']:7.3f}")

    # CIRCULARITY. Selecting on new_mae + alpha*reference_mae and then reporting
    # BWT (a function of reference_mae) scores the rule on the same quantity it
    # optimised. Only alpha=0 is free of this: it selects on the recent-window
    # error -- the signal a deployed system actually has -- and is evaluated on
    # the reference window, which never enters the decision. alpha>0 is reported
    # as an optimistic bound, and labelled as such.
    print()
    for a in ALPHAS:
        sel_bwt, sel_plast, picks = [], [], []
        for (seed, cyc), grp in d.groupby(["seed", "cycle"]):
            g = grp.dropna(subset=["new_mae", "reference_mae"])
            # the deployed gate only ever promotes a non-regressing candidate
            gp = g[g.verdict == "PROMOTE"] if (g.verdict == "PROMOTE").any() else g
            if gp.empty:
                continue
            g = gp
            score = g.new_mae + a * g.reference_mae
            best = g.loc[score.idxmin()]
            picks.append(best.condition)
            if cyc in MID:
                sel_bwt.append(best.bwt)
            if cyc == d.cycle.max():
                sel_plast.append(best.new_mae)
        share = {s: 100 * picks.count(s) / len(picks) for s in STRATS}
        print(f"  oracle alpha={a:<6g} BWT_mid={np.mean(sel_bwt):+7.3f}  "
              f"final plasticity={np.mean(sel_plast):7.3f}  "
              f"picks: " + " ".join(f"{s[:5]}={share[s]:4.0f}%" for s in STRATS))
        rows.append(dict(strategy=f"oracle_alpha{a:g}", alpha=a,
                         ret_mid=float(np.mean(sel_bwt)), ret_all=np.nan,
                         plast=float(np.mean(sel_plast))))
    return rows


def paired_tests(d):
    """Paired per-seed comparison of mid-stream BWT, as in the paper."""
    from scipy.stats import wilcoxon, ttest_rel
    print("\n" + "=" * 72)
    print("(3) PAIRED CONTRASTS ON MID-STREAM BWT (per seed, cycles 3-8)")
    print("=" * 72)
    m = (d[d.cycle.isin(MID)].groupby(["condition", "seed"]).bwt.mean()
         .unstack(0))
    for a, b in combinations(STRATS, 2):
        if a not in m or b not in m:
            continue
        x, y = m[a].dropna(), m[b].dropna()
        i = x.index.intersection(y.index)
        x, y = x[i], y[i]
        diff = (x - y)
        try:
            t, pt = ttest_rel(x, y)
            _, pw = wilcoxon(x, y)
        except Exception:
            t = pt = pw = float("nan")
        same = int(np.sum(np.sign(diff) == np.sign(diff.mean())))
        print(f"  {a:12s} vs {b:12s}  d={diff.mean():+7.3f}  "
              f"t={t:+6.2f} p_t={pt:.4f}  p_wilcoxon={pw:.4f}  "
              f"same sign {same}/{len(diff)}")


def main():
    d = load()
    print(f"[m1] {len(d)} jobs | strategies={STRATS} "
          f"seeds={sorted(d.seed.unique())} cycles={sorted(d.cycle.unique())}\n")
    winner_variability(d)
    rows = oracle_selection(d)
    paired_tests(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[m1] wrote {OUT}")
    print("[m1] NOTE: the oracle is an upper bound, not a simulation of the "
          "deployed system -- chains were built independently.")


if __name__ == "__main__":
    main()
