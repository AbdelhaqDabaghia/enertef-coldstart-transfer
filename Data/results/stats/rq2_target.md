# Multi-seed statistics — metric `target_nrmse`, ref `B3`
_source: Data/results/rq2_rq3_source.csv | grouping: `baseline` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3** (ref) | 10 | 0.5117 | 0.0196 | [0.4977, 0.5257] |
| B0 | 10 | 1.0492 | 0.1441 | [0.9461, 1.1523] |
| B1 | 10 | 0.7718 | 0.0129 | [0.7625, 0.7810] |
| B4 | 10 | 0.5140 | 0.0201 | [0.4997, 0.5284] |
| B5 | 10 | 0.6995 | 0.0300 | [0.6780, 0.7210] |

_Context: pooled within-condition SD ≈ 0.0453 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 10 | +0.5375 | +0.5240 | +1.000 | +3.542 | 0.001953 | 0.007812 |
| B1 | 10 | +0.2600 | +0.2571 | +1.000 | +12.662 | 0.001953 | 0.007812 |
| B4 | 10 | +0.0023 | +0.0018 | +0.455 | +0.554 | 0.2324 | 0.2324 |
| B5 | 10 | +0.1878 | +0.1959 | +1.000 | +5.685 | 0.001953 | 0.007812 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| B0 | 10 | +0.5375 | [+0.4495, +0.6254] | 5.069e-07 | 1 | 1 | no |
| B1 | 10 | +0.2600 | [+0.2481, +0.2719] | 4.842e-12 | 1 | 1 | no |
| B4 | 10 | +0.0023 | [-0.0001, +0.0047] | 1.849e-08 | 1.367e-07 | 1.367e-07 | **YES** |
| B5 | 10 | +0.1878 | [+0.1686, +0.2069] | 4.764e-09 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
