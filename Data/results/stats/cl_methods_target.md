# Multi-seed statistics — metric `target_nrmse`, ref `B3`
_source: Data/results/cl_methods_bench.csv | grouping: `baseline` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3** (ref) | 10 | 0.5116 | 0.0212 | [0.4964, 0.5268] |
| AGEM | 10 | 0.5592 | 0.0169 | [0.5471, 0.5713] |
| B5 | 10 | 0.6994 | 0.0304 | [0.6777, 0.7212] |
| DERpp | 10 | 0.7813 | 0.0181 | [0.7683, 0.7942] |
| LwF | 10 | 0.6750 | 0.0261 | [0.6563, 0.6937] |
| MAS | 10 | 0.8021 | 0.0282 | [0.7819, 0.8222] |

_Context: pooled within-condition SD ≈ 0.0235 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGEM | 10 | +0.0476 | +0.0441 | +0.964 | +1.496 | 0.003906 | 0.009766 |
| B5 | 10 | +0.1878 | +0.1913 | +1.000 | +5.907 | 0.001953 | 0.009766 |
| DERpp | 10 | +0.2697 | +0.2607 | +1.000 | +11.129 | 0.001953 | 0.009766 |
| LwF | 10 | +0.1634 | +0.1704 | +1.000 | +4.108 | 0.001953 | 0.009766 |
| MAS | 10 | +0.2905 | +0.2964 | +1.000 | +7.205 | 0.001953 | 0.009766 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| AGEM | 10 | +0.0476 | [+0.0292, +0.0660] | 4.331e-05 | 0.9886 | 0.9886 | no |
| B5 | 10 | +0.1878 | [+0.1694, +0.2062] | 3.397e-09 | 1 | 1 | no |
| DERpp | 10 | +0.2697 | [+0.2556, +0.2837] | 1.573e-11 | 1 | 1 | no |
| LwF | 10 | +0.1634 | [+0.1403, +0.1864] | 7.206e-08 | 1 | 1 | no |
| MAS | 10 | +0.2905 | [+0.2671, +0.3138] | 7.943e-10 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
