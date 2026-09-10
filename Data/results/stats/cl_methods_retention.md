# Multi-seed statistics — metric `source_retention_nrmse`, ref `B3`
_source: Data/results/cl_methods_bench.csv | grouping: `baseline` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3** (ref) | 10 | 0.5578 | 0.0300 | [0.5363, 0.5792] |
| AGEM | 10 | 0.5573 | 0.0311 | [0.5351, 0.5795] |
| B5 | 10 | 0.4298 | 0.0297 | [0.4086, 0.4510] |
| DERpp | 10 | 0.6232 | 0.0323 | [0.6000, 0.6463] |
| LwF | 10 | 0.5861 | 0.0237 | [0.5691, 0.6030] |
| MAS | 10 | 0.5660 | 0.0354 | [0.5407, 0.5913] |

_Context: pooled within-condition SD ≈ 0.0304 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGEM | 10 | -0.0004 | +0.0116 | +0.091 | -0.009 | 0.8457 | 0.9844 |
| B5 | 10 | -0.1280 | -0.1252 | -1.000 | -2.782 | 0.001953 | 0.009766 |
| DERpp | 10 | +0.0654 | +0.0679 | +0.964 | +1.686 | 0.003906 | 0.01562 |
| LwF | 10 | +0.0283 | +0.0329 | +0.636 | +0.590 | 0.08398 | 0.252 |
| MAS | 10 | +0.0082 | +0.0267 | +0.273 | +0.212 | 0.4922 | 0.9844 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| AGEM | 10 | -0.0004 | [-0.0299, +0.0290] | 0.0065 | 0.005942 | 0.0065 | **YES** |
| B5 | 10 | -0.1280 | [-0.1546, -0.1013] | 0.9998 | 3.259e-07 | 0.9998 | no |
| DERpp | 10 | +0.0654 | [+0.0429, +0.0879] | 2.967e-06 | 0.8794 | 0.8794 | no |
| LwF | 10 | +0.0283 | [+0.0005, +0.0561] | 0.0002972 | 0.09294 | 0.09294 | no |
| MAS | 10 | +0.0082 | [-0.0143, +0.0307] | 0.0005272 | 0.003916 | 0.003916 | **YES** |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
