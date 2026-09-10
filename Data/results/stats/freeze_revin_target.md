# Multi-seed statistics — metric `target_nrmse`, ref `freeze0`
_source: Data/results/freeze_revin.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **freeze0** (ref) | 10 | 0.1145 | 0.0049 | [0.1110, 0.1181] |
| freeze1 | 10 | 0.1153 | 0.0042 | [0.1122, 0.1183] |
| freeze2 | 10 | 0.1133 | 0.0042 | [0.1102, 0.1163] |
| freeze3 | 10 | 0.1150 | 0.0042 | [0.1121, 0.1180] |
| freeze4 | 10 | 0.1161 | 0.0029 | [0.1140, 0.1182] |
| freeze5 | 10 | 0.1372 | 0.0013 | [0.1362, 0.1381] |
| replay_ref | 10 | 0.1335 | 0.0081 | [0.1277, 0.1393] |

_Context: pooled within-condition SD ≈ 0.0043 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `freeze0` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze1 | 10 | +0.0007 | +0.0008 | +0.345 | +0.303 | 0.375 | 0.75 |
| freeze2 | 10 | -0.0013 | -0.0009 | -0.491 | -0.439 | 0.1934 | 0.5801 |
| freeze3 | 10 | +0.0005 | +0.0010 | +0.236 | +0.180 | 0.5566 | 0.75 |
| freeze4 | 10 | +0.0016 | +0.0031 | +0.673 | +0.496 | 0.06445 | 0.2578 |
| freeze5 | 10 | +0.0226 | +0.0228 | +1.000 | +5.254 | 0.001953 | 0.01172 |
| replay_ref | 10 | +0.0190 | +0.0216 | +1.000 | +1.621 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze1 | 10 | +0.0007 | [-0.0007, +0.0021] | 2.516e-10 | 4.76e-10 | 4.76e-10 | **YES** |
| freeze2 | 10 | -0.0013 | [-0.0030, +0.0004] | 4.361e-09 | 1.387e-09 | 4.361e-09 | **YES** |
| freeze3 | 10 | +0.0005 | [-0.0011, +0.0021] | 1.202e-09 | 1.874e-09 | 1.874e-09 | **YES** |
| freeze4 | 10 | +0.0016 | [-0.0003, +0.0034] | 2.29e-09 | 9.131e-09 | 9.131e-09 | **YES** |
| freeze5 | 10 | +0.0226 | [+0.0202, +0.0251] | 8.538e-11 | 0.9581 | 0.9581 | no |
| replay_ref | 10 | +0.0190 | [+0.0122, +0.0257] | 1.16e-06 | 0.3919 | 0.3919 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
