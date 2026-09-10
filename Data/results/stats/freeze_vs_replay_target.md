# Multi-seed statistics — metric `target_nrmse`, ref `replay_ref`
_source: Data/results/freeze_sweep.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **replay_ref** (ref) | 10 | 0.6993 | 0.0298 | [0.6780, 0.7207] |
| freeze0 | 10 | 0.5130 | 0.0185 | [0.4998, 0.5262] |
| freeze1 | 10 | 0.5494 | 0.0272 | [0.5299, 0.5689] |
| freeze2 | 10 | 0.5253 | 0.0275 | [0.5056, 0.5450] |
| freeze3 | 10 | 0.5067 | 0.0247 | [0.4890, 0.5244] |
| freeze4 | 10 | 0.8055 | 0.0089 | [0.7991, 0.8118] |
| freeze5 | 10 | 0.8525 | 0.0041 | [0.8495, 0.8555] |

_Context: pooled within-condition SD ≈ 0.0201 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `replay_ref` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze0 | 10 | -0.1863 | -0.1931 | -1.000 | -6.136 | 0.001953 | 0.01172 |
| freeze1 | 10 | -0.1499 | -0.1668 | -1.000 | -3.562 | 0.001953 | 0.01172 |
| freeze2 | 10 | -0.1740 | -0.1786 | -1.000 | -3.613 | 0.001953 | 0.01172 |
| freeze3 | 10 | -0.1926 | -0.1925 | -1.000 | -4.139 | 0.001953 | 0.01172 |
| freeze4 | 10 | +0.1061 | +0.0991 | +1.000 | +3.422 | 0.001953 | 0.01172 |
| freeze5 | 10 | +0.1532 | +0.1489 | +1.000 | +5.091 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze0 | 10 | -0.1863 | [-0.2039, -0.1687] | 1 | 2.411e-09 | 1 | no |
| freeze1 | 10 | -0.1499 | [-0.1743, -0.1255] | 1 | 2.266e-07 | 1 | no |
| freeze2 | 10 | -0.1740 | [-0.2019, -0.1461] | 1 | 2.312e-07 | 1 | no |
| freeze3 | 10 | -0.1926 | [-0.2196, -0.1657] | 1 | 7.808e-08 | 1 | no |
| freeze4 | 10 | +0.1061 | [+0.0882, +0.1241] | 2.129e-07 | 1 | 1 | no |
| freeze5 | 10 | +0.1532 | [+0.1357, +0.1706] | 1.041e-08 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
