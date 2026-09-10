# Multi-seed statistics — metric `source_retention_nrmse`, ref `B3`
_source: Data/results/rq2_rq3_source.csv | grouping: `baseline` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3** (ref) | 10 | 0.5573 | 0.0301 | [0.5357, 0.5789] |
| B0 | 10 | 1.5455 | 0.0874 | [1.4829, 1.6080] |
| B1 | 10 | 0.6155 | 0.0120 | [0.6070, 0.6241] |
| B4 | 10 | 0.5592 | 0.0294 | [0.5381, 0.5802] |
| B5 | 10 | 0.4296 | 0.0302 | [0.4080, 0.4512] |

_Context: pooled within-condition SD ≈ 0.0379 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 10 | +0.9882 | +1.0253 | +1.000 | +8.993 | 0.001953 | 0.007812 |
| B1 | 10 | +0.0582 | +0.0687 | +0.927 | +1.524 | 0.005859 | 0.01172 |
| B4 | 10 | +0.0018 | +0.0023 | +0.673 | +0.771 | 0.06445 | 0.06445 |
| B5 | 10 | -0.1277 | -0.1302 | -1.000 | -2.683 | 0.001953 | 0.007812 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| B0 | 10 | +0.9882 | [+0.9245, +1.0519] | 1.288e-10 | 1 | 1 | no |
| B1 | 10 | +0.0582 | [+0.0361, +0.0804] | 4.443e-06 | 0.7437 | 0.7437 | no |
| B4 | 10 | +0.0018 | [+0.0005, +0.0032] | 7.599e-14 | 1.475e-13 | 1.475e-13 | **YES** |
| B5 | 10 | -0.1277 | [-0.1553, -0.1001] | 0.9997 | 4.417e-07 | 0.9997 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
