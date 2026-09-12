# Multi-seed statistics — metric `source_retention_nrmse`, ref `B5_replay`
_source: Data/results/cl_revin.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B5_replay** (ref) | 10 | 0.2051 | 0.0104 | [0.1976, 0.2125] |
| AGEM | 10 | 0.2232 | 0.0060 | [0.2189, 0.2274] |
| B3_warm | 10 | 0.2366 | 0.0058 | [0.2325, 0.2408] |
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

## Paired contrast vs `B5_replay` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGEM | 10 | +0.0181 | +0.0190 | +1.000 | +2.304 | 0.001953 | 0.02344 |
| B3_warm | 10 | +0.0316 | +0.0333 | +1.000 | +3.020 | 0.001953 | 0.02344 |
| DERpp | 10 | +0.0392 | +0.0402 | +1.000 | +2.986 | 0.001953 | 0.02344 |
| LwF | 10 | +0.0747 | +0.0729 | +1.000 | +6.342 | 0.001953 | 0.02344 |
| MAS | 10 | -0.0104 | -0.0072 | -0.855 | -0.978 | 0.01367 | 0.1094 |
| V1_empFisher_l1 | 10 | -0.0053 | -0.0026 | -0.564 | -0.463 | 0.1309 | 0.3926 |
| V1_empFisher_l10 | 10 | -0.0119 | -0.0104 | -0.855 | -1.009 | 0.01367 | 0.1094 |
| V1_empFisher_l100 | 10 | -0.0104 | -0.0100 | -0.782 | -0.928 | 0.02734 | 0.1641 |
| V2_perlayer_l1 | 10 | -0.0043 | -0.0015 | -0.309 | -0.357 | 0.4316 | 0.8633 |
| V2_perlayer_l10 | 10 | -0.0040 | -0.0017 | -0.273 | -0.372 | 0.4922 | 0.8633 |
| V2_perlayer_l100 | 10 | -0.0068 | -0.0047 | -0.745 | -0.659 | 0.03711 | 0.1855 |
| V3_hybrid | 10 | -0.0061 | -0.0066 | -0.745 | -0.791 | 0.03711 | 0.1855 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| AGEM | 10 | +0.0181 | [+0.0135, +0.0226] | 2.766e-10 | 2.148e-07 | 2.148e-07 | **YES** |
| B3_warm | 10 | +0.0316 | [+0.0255, +0.0376] | 7.085e-10 | 0.0001733 | 0.0001733 | **YES** |
| DERpp | 10 | +0.0392 | [+0.0316, +0.0468] | 2.41e-09 | 0.01437 | 0.01437 | **YES** |
| LwF | 10 | +0.0747 | [+0.0678, +0.0815] | 4.652e-11 | 1 | 1 | no |
| MAS | 10 | -0.0104 | [-0.0165, -0.0042] | 4.341e-07 | 1.136e-08 | 4.341e-07 | **YES** |
| V1_empFisher_l1 | 10 | -0.0053 | [-0.0120, +0.0013] | 3.103e-07 | 4.935e-08 | 3.103e-07 | **YES** |
| V1_empFisher_l10 | 10 | -0.0119 | [-0.0187, -0.0051] | 1.464e-06 | 2.3e-08 | 1.464e-06 | **YES** |
| V1_empFisher_l100 | 10 | -0.0104 | [-0.0168, -0.0039] | 6.842e-07 | 1.817e-08 | 6.842e-07 | **YES** |
| V2_perlayer_l1 | 10 | -0.0043 | [-0.0112, +0.0027] | 3.583e-07 | 8.224e-08 | 3.583e-07 | **YES** |
| V2_perlayer_l10 | 10 | -0.0040 | [-0.0103, +0.0023] | 1.53e-07 | 3.761e-08 | 1.53e-07 | **YES** |
| V2_perlayer_l100 | 10 | -0.0068 | [-0.0128, -0.0008] | 1.683e-07 | 1.558e-08 | 1.683e-07 | **YES** |
| V3_hybrid | 10 | -0.0061 | [-0.0106, -0.0016] | 1.165e-08 | 1.333e-09 | 1.165e-08 | **YES** |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
