# Multi-seed statistics — metric `source_retention_nrmse`, ref `replay_ref`
_source: Data/results/freeze_sweep.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **replay_ref** (ref) | 10 | 0.4293 | 0.0296 | [0.4082, 0.4505] |
| freeze0 | 10 | 0.5572 | 0.0284 | [0.5369, 0.5775] |
| freeze1 | 10 | 0.5337 | 0.0350 | [0.5086, 0.5588] |
| freeze2 | 10 | 0.4816 | 0.0323 | [0.4586, 0.5047] |
| freeze3 | 10 | 0.4196 | 0.0228 | [0.4033, 0.4360] |
| freeze4 | 10 | 0.6261 | 0.0103 | [0.6188, 0.6335] |
| freeze5 | 10 | 0.8021 | 0.0091 | [0.7956, 0.8086] |

_Context: pooled within-condition SD ≈ 0.0239 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `replay_ref` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze0 | 10 | +0.1279 | +0.1293 | +1.000 | +2.852 | 0.001953 | 0.01172 |
| freeze1 | 10 | +0.1044 | +0.1185 | +1.000 | +1.963 | 0.001953 | 0.01172 |
| freeze2 | 10 | +0.0523 | +0.0476 | +0.818 | +1.012 | 0.01953 | 0.03906 |
| freeze3 | 10 | -0.0097 | -0.0174 | -0.236 | -0.216 | 0.5566 | 0.5566 |
| freeze4 | 10 | +0.1968 | +0.2043 | +1.000 | +5.826 | 0.001953 | 0.01172 |
| freeze5 | 10 | +0.3728 | +0.3771 | +1.000 | +11.498 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze0 | 10 | +0.1279 | [+0.1019, +0.1539] | 2.631e-07 | 0.9998 | 0.9998 | no |
| freeze1 | 10 | +0.1044 | [+0.0736, +0.1352] | 3.63e-06 | 0.9949 | 0.9949 | no |
| freeze2 | 10 | +0.0523 | [+0.0223, +0.0823] | 7.418e-05 | 0.5546 | 0.5546 | no |
| freeze3 | 10 | -0.0097 | [-0.0357, +0.0163] | 0.009645 | 0.001137 | 0.009645 | **YES** |
| freeze4 | 10 | +0.1968 | [+0.1772, +0.2164] | 1.267e-09 | 1 | 1 | no |
| freeze5 | 10 | +0.3728 | [+0.3540, +0.3916] | 7.225e-12 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
