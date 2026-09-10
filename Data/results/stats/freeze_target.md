# Multi-seed statistics — metric `target_nrmse`, ref `freeze0`
_source: Data/results/freeze_sweep.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **freeze0** (ref) | 10 | 0.5130 | 0.0185 | [0.4998, 0.5262] |
| freeze1 | 10 | 0.5494 | 0.0272 | [0.5299, 0.5689] |
| freeze2 | 10 | 0.5253 | 0.0275 | [0.5056, 0.5450] |
| freeze3 | 10 | 0.5067 | 0.0247 | [0.4890, 0.5244] |
| freeze4 | 10 | 0.8055 | 0.0089 | [0.7991, 0.8118] |
| freeze5 | 10 | 0.8525 | 0.0041 | [0.8495, 0.8555] |
| replay_ref | 10 | 0.6993 | 0.0298 | [0.6780, 0.7207] |

_Context: pooled within-condition SD ≈ 0.0201 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `freeze0` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| freeze1 | 10 | +0.0364 | +0.0287 | +1.000 | +1.624 | 0.001953 | 0.01172 |
| freeze2 | 10 | +0.0123 | +0.0119 | +0.455 | +0.423 | 0.2324 | 0.4648 |
| freeze3 | 10 | -0.0063 | -0.0035 | -0.273 | -0.220 | 0.4922 | 0.4922 |
| freeze4 | 10 | +0.2925 | +0.2905 | +1.000 | +14.597 | 0.001953 | 0.01172 |
| freeze5 | 10 | +0.3395 | +0.3445 | +1.000 | +17.979 | 0.001953 | 0.01172 |
| replay_ref | 10 | +0.1863 | +0.1931 | +1.000 | +6.136 | 0.001953 | 0.01172 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| freeze1 | 10 | +0.0364 | [+0.0234, +0.0494] | 1.156e-05 | 0.9771 | 0.9771 | no |
| freeze2 | 10 | +0.0123 | [-0.0046, +0.0292] | 0.003324 | 0.2129 | 0.2129 | no |
| freeze3 | 10 | -0.0063 | [-0.0229, +0.0103] | 0.08219 | 0.008703 | 0.08219 | no |
| freeze4 | 10 | +0.2925 | [+0.2809, +0.3041] | 1.453e-12 | 1 | 1 | no |
| freeze5 | 10 | +0.3395 | [+0.3286, +0.3504] | 2.426e-13 | 1 | 1 | no |
| replay_ref | 10 | +0.1863 | [+0.1687, +0.2039] | 2.411e-09 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
