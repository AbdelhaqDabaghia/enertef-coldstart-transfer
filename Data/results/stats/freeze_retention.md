# Multi-seed statistics — metric `source_retention_nrmse`, ref `freeze0`
_source: Data/results/freeze_sweep.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **freeze0** (ref) | 10 | 0.5572 | 0.0284 | [0.5369, 0.5775] |
| freeze1 | 10 | 0.5337 | 0.0350 | [0.5086, 0.5588] |
| freeze2 | 10 | 0.4816 | 0.0323 | [0.4586, 0.5047] |
| freeze3 | 10 | 0.4196 | 0.0228 | [0.4033, 0.4360] |
| freeze4 | 10 | 0.6261 | 0.0103 | [0.6188, 0.6335] |
| freeze5 | 10 | 0.8021 | 0.0091 | [0.7956, 0.8086] |
| replay_ref | 10 | 0.4293 | 0.0296 | [0.4082, 0.4505] |

_Context: pooled within-condition SD ≈ 0.0239 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `freeze0` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze1 | 10 | -0.0235 | -0.0284 | -0.891 | -1.286 | 0.009766 | 0.01172 |
| freeze2 | 10 | -0.0756 | -0.0780 | -1.000 | -2.354 | 0.001953 | 0.01172 |
| freeze3 | 10 | -0.1376 | -0.1353 | -1.000 | -4.585 | 0.001953 | 0.01172 |
| freeze4 | 10 | +0.0689 | +0.0744 | +1.000 | +3.353 | 0.001953 | 0.01172 |
| freeze5 | 10 | +0.2449 | +0.2501 | +1.000 | +11.818 | 0.001953 | 0.01172 |
| replay_ref | 10 | -0.1279 | -0.1293 | -1.000 | -2.852 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze1 | 10 | -0.0235 | [-0.0341, -0.0129] | 0.0006569 | 2.339e-07 | 0.0006569 | **YES** |
| freeze2 | 10 | -0.0756 | [-0.0942, -0.0570] | 0.9836 | 2.97e-07 | 0.9836 | no |
| freeze3 | 10 | -0.1376 | [-0.1550, -0.1202] | 1 | 5.031e-09 | 1 | no |
| freeze4 | 10 | +0.0689 | [+0.0570, +0.0809] | 9.968e-09 | 0.9914 | 0.9914 | no |
| freeze5 | 10 | +0.2449 | [+0.2329, +0.2569] | 3.304e-12 | 1 | 1 | no |
| replay_ref | 10 | -0.1279 | [-0.1539, -0.1019] | 0.9998 | 2.631e-07 | 0.9998 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
