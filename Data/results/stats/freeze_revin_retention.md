# Multi-seed statistics — metric `source_retention_nrmse`, ref `freeze0`
_source: Data/results/freeze_revin.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **freeze0** (ref) | 10 | 0.2354 | 0.0038 | [0.2326, 0.2381] |
| freeze1 | 10 | 0.2315 | 0.0035 | [0.2289, 0.2340] |
| freeze2 | 10 | 0.2316 | 0.0042 | [0.2286, 0.2347] |
| freeze3 | 10 | 0.2318 | 0.0029 | [0.2298, 0.2339] |
| freeze4 | 10 | 0.2315 | 0.0026 | [0.2296, 0.2333] |
| freeze5 | 10 | 0.1967 | 0.0024 | [0.1950, 0.1984] |
| replay_ref | 10 | 0.2093 | 0.0115 | [0.2011, 0.2175] |

_Context: pooled within-condition SD ≈ 0.0044 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `freeze0` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze1 | 10 | -0.0039 | -0.0034 | -1.000 | -2.278 | 0.001953 | 0.01172 |
| freeze2 | 10 | -0.0037 | -0.0030 | -1.000 | -2.176 | 0.001953 | 0.01172 |
| freeze3 | 10 | -0.0036 | -0.0029 | -1.000 | -1.663 | 0.001953 | 0.01172 |
| freeze4 | 10 | -0.0039 | -0.0035 | -0.927 | -1.163 | 0.005859 | 0.01172 |
| freeze5 | 10 | -0.0386 | -0.0381 | -1.000 | -7.807 | 0.001953 | 0.01172 |
| replay_ref | 10 | -0.0261 | -0.0264 | -1.000 | -2.239 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze1 | 10 | -0.0039 | [-0.0049, -0.0029] | 1.066e-14 | 2.618e-15 | 1.066e-14 | **YES** |
| freeze2 | 10 | -0.0037 | [-0.0047, -0.0027] | 1.056e-14 | 2.753e-15 | 1.056e-14 | **YES** |
| freeze3 | 10 | -0.0036 | [-0.0048, -0.0023] | 7.359e-14 | 2.048e-14 | 7.359e-14 | **YES** |
| freeze4 | 10 | -0.0039 | [-0.0058, -0.0020] | 4.355e-12 | 1.077e-12 | 4.355e-12 | **YES** |
| freeze5 | 10 | -0.0386 | [-0.0415, -0.0358] | 2.368e-05 | 4.194e-13 | 2.368e-05 | **YES** |
| replay_ref | 10 | -0.0261 | [-0.0329, -0.0193] | 5.694e-05 | 3.436e-09 | 5.694e-05 | **YES** |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
