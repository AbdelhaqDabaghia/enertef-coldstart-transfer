"""
stats_report.py -- paper-ready statistics for the multi-seed CL campaign.

Reads the per-run result CSV(s) written by the drivers (logger.FIELDS schema) and
produces, for one metric and a chosen reference condition:

  * descriptives    : n_seeds, mean, std (ddof=1), 95% t-CI
  * paired contrast : seed-matched paired Wilcoxon signed-rank vs. the reference,
                      median & mean difference, matched-pairs rank-biserial and
                      Cohen's d_z effect sizes, Holm-Bonferroni-corrected p
  * equivalence     : paired TOST at a PRE-DECLARED margin delta (the correct tool
                      for the negative "EWC adds nothing over warm-start" claim --
                      a non-significant difference test is NOT evidence of
                      equivalence). Reported as the 90% CI of the mean difference
                      and max(p_lower, p_upper); "equivalent" iff that CI lies
                      entirely within [-delta, +delta].

No GPU, no TensorFlow: pure pandas/numpy/scipy over the logged CSVs, so it is
deterministic and re-runnable anywhere. Statsmodels is NOT required.

Examples
--------
# EWC bench: is every EWC variant equivalent to warm-start on target nRMSE?
python -m coldstart_transfer.stats_report \
    --csv Data/results/ewc_bench.csv --group notes --ref B3_warm \
    --metric target_nrmse --tost-margin 0.02 \
    --out Data/results/stats_ewc_bench_target.md

# Source-retention (stability): does replay beat warm-start? (expect NOT equivalent)
python -m coldstart_transfer.stats_report \
    --csv Data/results/ewc_bench.csv --group notes --ref B3_warm \
    --metric source_retention_nrmse --tost-margin 0.05

# RQ2/RQ3 conditions keyed by the `baseline` column:
python -m coldstart_transfer.stats_report \
    --csv Data/results/rq2_rq3_source.csv --group baseline --ref B3 \
    --metric target_nrmse --tost-margin 0.02

Choosing --tost-margin (delta): it must be fixed A PRIORI on domain grounds, not
read off the data. A defensible choice is the smallest nRMSE difference that would
change an operational decision (e.g. 0.02 = 2 percentage-points of relative error).
The tool prints the pooled within-condition SD as *context* for sanity, never as
the margin itself.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

# lower-is-better for every metric we log (nRMSE, RMSE); used only for the
# human-readable "better/worse" annotation, never for the statistics themselves.
LOWER_IS_BETTER = True


def _load(csvs: list[str], group: str, metric: str) -> pd.DataFrame:
    frames = []
    for c in csvs:
        df = pd.read_csv(c)
        if group not in df.columns:
            sys.exit(f"[stats] {c}: no '{group}' column (have: {list(df.columns)})")
        if metric not in df.columns:
            sys.exit(f"[stats] {c}: no '{metric}' column (have: {list(df.columns)})")
        if "seed" not in df.columns:
            sys.exit(f"[stats] {c}: no 'seed' column")
        frames.append(df[[group, "seed", metric]].copy())
    df = pd.concat(frames, ignore_index=True)
    # a blank group cell (driver_source logs empty notes) falls back to nothing:
    # caller should pick --group baseline for those files.
    df[group] = df[group].astype(str).str.strip()
    df = df[df[group] != ""]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    return df


def _pivot(df: pd.DataFrame, group: str, metric: str) -> pd.DataFrame:
    """condition x seed matrix of the metric (mean if a (cond,seed) repeats)."""
    return df.pivot_table(index=group, columns="seed", values=metric, aggfunc="mean")


def _ci95(x: np.ndarray) -> tuple[float, float, float, float]:
    n = len(x)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    if n > 1:
        sem = sd / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * sem
    else:
        h = float("nan")
    return m, sd, m - h, m + h


def _paired_wilcoxon(diff: np.ndarray):
    """Two-sided paired Wilcoxon; returns (stat, p, rank_biserial). Handles the
    all-zero / all-equal degenerate cases scipy would otherwise raise on."""
    nz = diff[diff != 0]
    if len(nz) == 0:
        return float("nan"), 1.0, 0.0
    try:
        res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided",
                             correction=False, mode="auto")
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    # matched-pairs rank-biserial correlation from signed ranks
    r = stats.rankdata(np.abs(nz))
    rp = r[nz > 0].sum()
    rm = r[nz < 0].sum()
    tot = rp + rm
    rbc = float((rp - rm) / tot) if tot > 0 else 0.0
    return stat, p, rbc


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN-safe)."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for k, i in enumerate(order):
        val = (m - k) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def _tost(diff: np.ndarray, delta: float, alpha: float = 0.05):
    """Paired TOST (two one-sided t-tests) for equivalence within +/- delta.
    Equivalence <=> the (1-2*alpha) CI of the mean difference lies in [-d, d].
    Returns (mean_d, lo90, hi90, p_lower, p_upper, p_tost, equivalent)."""
    n = len(diff)
    md = float(np.mean(diff))
    if n < 2:
        return md, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False
    sd = float(np.std(diff, ddof=1))
    sem = sd / np.sqrt(n)
    if sem == 0:
        eq = abs(md) < delta
        return md, md, md, 0.0 if eq else 1.0, 0.0 if eq else 1.0, 0.0 if eq else 1.0, eq
    dfree = n - 1
    # H0a: mean <= -delta  (test that mean > -delta)
    t_lower = (md + delta) / sem
    p_lower = stats.t.sf(t_lower, dfree)         # P(T > t_lower)
    # H0b: mean >= +delta  (test that mean < +delta)
    t_upper = (md - delta) / sem
    p_upper = stats.t.cdf(t_upper, dfree)        # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    tcrit = stats.t.ppf(1 - alpha, dfree)        # one-sided crit -> (1-2a) CI
    lo = md - tcrit * sem
    hi = md + tcrit * sem
    equivalent = (lo > -delta) and (hi < delta)
    return md, lo, hi, p_lower, p_upper, p_tost, equivalent


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paper-ready multi-seed statistics.")
    ap.add_argument("--csv", nargs="+", required=True, help="result CSV(s)")
    ap.add_argument("--group", default="baseline",
                    help="condition column: 'baseline' (driver_source) or 'notes' (ewc_bench)")
    ap.add_argument("--metric", default="target_nrmse",
                    help="target_nrmse | source_retention_nrmse | target_rmse_kw")
    ap.add_argument("--ref", required=True, help="reference condition (e.g. B3 or B3_warm)")
    ap.add_argument("--tost-margin", type=float, default=None,
                    help="pre-declared equivalence margin delta (same units as metric)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args(argv)

    # markdown uses δ/Δ/±/≈; keep them on a cp1252 Windows console too
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    df = _load(args.csv, args.group, args.metric)
    mat = _pivot(df, args.group, args.metric)
    if args.ref not in mat.index:
        sys.exit(f"[stats] ref '{args.ref}' not among conditions: {list(mat.index)}")

    ref_row = mat.loc[args.ref]
    conds = [c for c in mat.index if c != args.ref]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Multi-seed statistics — metric `{args.metric}`, ref `{args.ref}`")
    emit(f"_source: {', '.join(args.csv)} | grouping: `{args.group}` | "
         f"lower is {'better' if LOWER_IS_BETTER else 'worse'}_\n")

    # ---- descriptives ----
    emit("## Descriptives")
    emit("| condition | n | mean | std | 95% CI |")
    emit("|---|---:|---:|---:|---|")
    for c in [args.ref] + conds:
        x = mat.loc[c].dropna().to_numpy()
        m, sd, lo, hi = _ci95(x)
        emit(f"| {'**'+c+'** (ref)' if c==args.ref else c} | {len(x)} | "
             f"{m:.4f} | {sd:.4f} | [{lo:.4f}, {hi:.4f}] |")
    pooled_sd = float(np.nanmean([mat.loc[c].dropna().std(ddof=1)
                                  for c in mat.index if mat.loc[c].dropna().size > 1]))
    emit(f"\n_Context: pooled within-condition SD ≈ {pooled_sd:.4f} "
         f"(a sanity reference for choosing δ — NOT the margin itself)._\n")

    # ---- paired contrasts vs ref ----
    emit(f"## Paired contrast vs `{args.ref}` (seed-matched)")
    emit("| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    raw_p, rows = [], []
    for c in conds:
        pair = pd.concat([ref_row.rename("ref"), mat.loc[c].rename("cond")], axis=1).dropna()
        seeds = pair.index.to_numpy()
        diff = (pair["cond"] - pair["ref"]).to_numpy()  # cond - ref
        stat, p, rbc = _paired_wilcoxon(diff)
        dz = float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else float("nan")
        raw_p.append(p)
        rows.append((c, len(diff), float(np.mean(diff)), float(np.median(diff)), rbc, dz, p))
    holm = _holm(raw_p)
    for (c, n, dm, dmed, rbc, dz, p), ph in zip(rows, holm):
        emit(f"| {c} | {n} | {dm:+.4f} | {dmed:+.4f} | {rbc:+.3f} | "
             f"{dz:+.3f} | {p:.4g} | {ph:.4g} |")
    emit("\n_Δ = condition − ref; Δ<0 means the condition is better (lower error). "
         "Holm corrects across the whole family of contrasts shown._\n")

    # ---- equivalence (TOST) ----
    if args.tost_margin is None:
        emit("## Equivalence (TOST)\n_Skipped: pass `--tost-margin δ` (pre-declared) "
             "to test the negative 'adds nothing' claim properly._")
    else:
        d = args.tost_margin
        emit(f"## Equivalence — paired TOST at δ = ±{d:g} (α = {args.alpha})")
        emit("| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |")
        emit("|---|---:|---:|---|---:|---:|---:|:--:|")
        for c in conds:
            pair = pd.concat([ref_row.rename("ref"), mat.loc[c].rename("cond")], axis=1).dropna()
            diff = (pair["cond"] - pair["ref"]).to_numpy()
            md, lo, hi, pl, pu, pt, eq = _tost(diff, d, args.alpha)
            emit(f"| {c} | {len(diff)} | {md:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                 f"{pl:.4g} | {pu:.4g} | {pt:.4g} | {'**YES**' if eq else 'no'} |")
        emit(f"\n_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−{d:g}, +{d:g}]. "
             "This — not a non-significant difference test — is what licenses "
             "'condition X adds nothing over the reference'._")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[stats] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
