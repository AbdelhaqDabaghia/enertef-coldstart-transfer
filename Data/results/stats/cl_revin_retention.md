# Multi-seed statistics — metric `source_retention_nrmse`, ref `B3_warm`
_source: Data/results/cl_revin.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3_warm** (ref) | 10 | 0.2366 | 0.0058 | [0.2325, 0.2408] |
| AGEM | 10 | 0.2232 | 0.0060 | [0.2189, 0.2274] |
| B5_replay | 10 | 0.2051 | 0.0104 | [0.1976, 0.2125] |
| DERpp | 10 | 0.2443 | 0.0114 | [0.2361, 0.2524] |
| LwF | 10 | 0.2797 | 0.0057 | [0.2757, 0.2838] |
| MAS | 10 | 0.1947 | 0.0025 | [0.1929, 0.1965] |
| V1_empFisher_l1 | 10 | 0.1997 | 0.0054 | [0.1959, 0.2036] |
| V1_empFisher_l10 | 10 | 0.1932 | 0.0028 | [0.1911, 0.1952] |
| V1_empFisher_l100 | 10 | 0.1947 | 0.0027 | [0.1928, 0.1966] |
| V2_perlayer_l1 | 10 | 0.2008 | 0.0059 | [0.1965, 0.2050] |
| V2_perlayer_l10 | 10 | 0.2010 | 0.0041 | [0.1981, 0.2039] |
| V2_perlayer_l100 | 10 | 0.1983 | 0.0019 | [0.1969, 0.1996] |
| V3_hybrid | 10 | 0.1989 | 0.0036 | [0.1964, 0.2015] |

_Context: pooled within-condition SD ≈ 0.0052 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3_warm` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGEM | 10 | -0.0135 | -0.0146 | -0.964 | -1.653 | 0.003906 | 0.02344 |
| B5_replay | 10 | -0.0316 | -0.0333 | -1.000 | -3.020 | 0.001953 | 0.02344 |
| DERpp | 10 | +0.0076 | +0.0076 | +0.564 | +0.577 | 0.1309 | 0.1309 |
| LwF | 10 | +0.0431 | +0.0423 | +1.000 | +5.732 | 0.001953 | 0.02344 |
| MAS | 10 | -0.0419 | -0.0415 | -1.000 | -7.750 | 0.001953 | 0.02344 |
| V1_empFisher_l1 | 10 | -0.0369 | -0.0363 | -1.000 | -6.787 | 0.001953 | 0.02344 |
| V1_empFisher_l10 | 10 | -0.0435 | -0.0441 | -1.000 | -6.788 | 0.001953 | 0.02344 |
| V1_empFisher_l100 | 10 | -0.0419 | -0.0414 | -1.000 | -6.622 | 0.001953 | 0.02344 |
| V2_perlayer_l1 | 10 | -0.0358 | -0.0369 | -1.000 | -6.552 | 0.001953 | 0.02344 |
| V2_perlayer_l10 | 10 | -0.0356 | -0.0359 | -1.000 | -6.889 | 0.001953 | 0.02344 |
| V2_perlayer_l100 | 10 | -0.0384 | -0.0371 | -1.000 | -6.599 | 0.001953 | 0.02344 |
| V3_hybrid | 10 | -0.0377 | -0.0371 | -1.000 | -6.434 | 0.001953 | 0.02344 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| AGEM | 10 | -0.0135 | [-0.0182, -0.0088] | 9.269e-08 | 7.21e-10 | 9.269e-08 | **YES** |
| B5_replay | 10 | -0.0316 | [-0.0376, -0.0255] | 0.0001733 | 7.085e-10 | 0.0001733 | **YES** |
| DERpp | 10 | +0.0076 | [-0.0000, +0.0153] | 1.177e-07 | 1.606e-06 | 1.606e-06 | **YES** |
| LwF | 10 | +0.0431 | [+0.0387, +0.0474] | 1.148e-11 | 0.008648 | 0.008648 | **YES** |
| MAS | 10 | -0.0419 | [-0.0451, -0.0388] | 0.0005524 | 6.739e-13 | 0.0005524 | **YES** |
| V1_empFisher_l1 | 10 | -0.0369 | [-0.0400, -0.0337] | 1.612e-05 | 1.163e-12 | 1.612e-05 | **YES** |
| V1_empFisher_l10 | 10 | -0.0435 | [-0.0472, -0.0397] | 0.005133 | 2.629e-12 | 0.005133 | **YES** |
| V1_empFisher_l100 | 10 | -0.0419 | [-0.0456, -0.0383] | 0.001503 | 2.765e-12 | 0.001503 | **YES** |
| V2_perlayer_l1 | 10 | -0.0358 | [-0.0390, -0.0327] | 9.251e-06 | 1.376e-12 | 9.251e-06 | **YES** |
| V2_perlayer_l10 | 10 | -0.0356 | [-0.0386, -0.0326] | 5.137e-06 | 8.479e-13 | 5.137e-06 | **YES** |
| V2_perlayer_l100 | 10 | -0.0384 | [-0.0418, -0.0350] | 6.902e-05 | 1.833e-12 | 6.902e-05 | **YES** |
| V3_hybrid | 10 | -0.0377 | [-0.0411, -0.0343] | 4.703e-05 | 2.097e-12 | 4.703e-05 | **YES** |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
