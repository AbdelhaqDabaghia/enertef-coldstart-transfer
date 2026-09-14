"""
e7_compare.py -- do the paper's conclusions survive the causal pipeline?

Reads the archived (leaky) and corrected (causal) results side by side and
re-runs every statistical test the manuscript relies on, under both. The point
is not which pipeline gives better numbers -- the causal one gives worse ones by
construction -- but whether the ORDERINGS and the SIGNIFICANCE hold.

Claims under test, each with the section of journal_v3 that makes it:

  C1  Sec. V.C   OSR is indistinguishable from fine-tuning on raw retention and
                 better once gauged. If the gauged advantage vanishes causally,
                 the gauge decomposition contribution goes with it.
  C2  Sec. V.C   Replay keeps ~70 % of its advantage after gauge correction.
  C3  Sec. V.B   Gauge correction helps at all, i.e. gauged < raw. The
                 reference check already shows this FAILS for the un-adapted
                 model causally (1.2542 -> 1.2584); the question is whether it
                 also fails after adaptation.
  C4  Sec.VIII.A Ordering by forecast error differs from ordering by realised
                 controller cost.
  C5  Sec.VII.A  Per-day oracle selection recovers almost nothing.

Writes Data/results/corrected_causal_pipeline/e7_comparison.md.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LEAKY = "Data/results"
CAUSAL = "Data/results/corrected_causal_pipeline"
OUT = os.environ.get("OUT", os.path.join(CAUSAL, "e7_comparison.md"))
CONDS = ["B3_warm", "MAS", "B5_replay"]


def _load(d, name):
    p = os.path.join(d, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _pair(df, col, a, b):
    """Seed-matched difference a-b with a Wilcoxon test."""
    p = df[df.seed >= 0].pivot(index="seed", columns="condition", values=col)
    if a not in p or b not in p:
        return None
    x, y = p[a].dropna(), p[b].dropna()
    i = x.index.intersection(y.index)
    x, y = x.loc[i], y.loc[i]
    if len(x) < 3:
        return None
    d = x - y
    try:
        _, pv = wilcoxon(x, y)
    except ValueError:                      # all differences zero
        pv = 1.0
    return dict(n=len(x), mean=float(d.mean()), p=float(pv),
                dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                a=float(x.mean()), b=float(y.mean()))


def gauge_block(lines, tag, df):
    lines.append("\n### %s pipeline -- e13 gauge decomposition\n" % tag)
    if df is None:
        lines.append("_not available_\n")
        return
    g = df[df.seed >= 0].groupby("condition")[["raw", "gauged", "a"]].mean()
    ref = df[df.seed < 0]
    lines.append("| condition | raw | gauged | a | gauged helps? |")
    lines.append("|---|---|---|---|---|")
    if len(ref):
        r = ref.iloc[0]
        lines.append("| reference (no adaptation) | %.4f | %.4f | %.3f | %s |"
                     % (r["raw"], r["gauged"], r["a"],
                        "yes" if r["gauged"] < r["raw"] else "**NO**"))
    for c in CONDS:
        if c in g.index:
            row = g.loc[c]
            lines.append("| %s | %.4f | %.4f | %.3f | %s |"
                         % (c, row["raw"], row["gauged"], row["a"],
                            "yes" if row["gauged"] < row["raw"] else "**NO**"))

    lines.append("\n**C1 / C2 -- paired tests**\n")
    lines.append("| metric | contrast | mean diff | p | dz | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for col in ["raw", "gauged"]:
        for a, b in [("MAS", "B3_warm"), ("B5_replay", "B3_warm")]:
            r = _pair(df, col, a, b)
            if r:
                lines.append("| %s | %s vs %s | %+.4f | %.4f | %+.2f | %s |"
                             % (col, a, b, r["mean"], r["p"], r["dz"],
                                "better" if r["mean"] < 0 and r["p"] < 0.05
                                else ("worse" if r["mean"] > 0 and r["p"] < 0.05
                                      else "n.s.")))


def decision_block(lines, tag, df):
    lines.append("\n### %s pipeline -- e16 accuracy vs decision cost\n" % tag)
    if df is None:
        lines.append("_not available_\n")
        return
    o = df.pivot_table(index=["seed", "day"], columns="condition", values="objective")
    m = df.pivot_table(index=["seed", "day"], columns="condition", values="mae")
    lines.append("| condition | MAE (kW) | realised objective |")
    lines.append("|---|---|---|")
    for c in o.columns:
        lines.append("| %s | %.2f | %.1f |" % (c, m[c].mean(), o[c].mean()))

    best_mae = m.mean().idxmin()
    best_obj = o.mean().idxmin()
    lines.append("\n- best by MAE: **%s** | best by controller cost: **%s**"
                 % (best_mae, best_obj))
    lines.append("- **C4 inversion: %s**"
                 % ("HOLDS -- the two disagree" if best_mae != best_obj
                    else "FAILS -- the same model wins both"))
    orc = o.min(axis=1).mean()
    bf = o[best_obj].mean()
    lines.append("- C5 oracle ceiling over best fixed: %+.2f (%+.2f %%), n=%d"
                 % (orc - bf, 100 * (orc - bf) / bf, len(o)))


def main():
    lines = ["# E7 -- does the paper survive causal features?",
             "",
             "Generated by `scripts/e7_compare.py`. The causal arm uses",
             "`*_causal.csv` features, in which every feature at time t depends",
             "only on y_(t-1) and earlier (verified by `tests/test_causality.py`).",
             ""]

    ref = _load(CAUSAL, "e7_reference_check.csv")
    if ref is not None:
        lines.append("## The un-adapted production model\n")
        lines.append(ref.to_markdown(index=False))
        lines.append("")

    for tag, d in [("LEAKY (archived)", LEAKY), ("CAUSAL (corrected)", CAUSAL)]:
        gauge_block(lines, tag, _load(d, "e13_gauge_decomposition.csv"))
        decision_block(lines, tag, _load(d, "e16_selection_oracle.csv"))

    lines += ["", "## How to read this",
              "",
              "The causal numbers are worse in absolute terms by construction --",
              "the leak was supplying the target. That is not the question. The",
              "question is whether the ORDERINGS and the SIGNIFICANCE survive.",
              "Any row whose verdict changes between the two pipelines is a claim",
              "the manuscript cannot make without the qualification that it holds",
              "only under a feature construction the deployed service does not use.",
              ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\n[e7_compare] wrote %s" % OUT)


if __name__ == "__main__":
    main()
