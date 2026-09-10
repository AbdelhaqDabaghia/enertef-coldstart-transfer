# Multi-seed statistics — metric `source_retention_nrmse`, ref `B3_warm`
_source: Data/results/ewc_bench.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3_warm** (ref) | 10 | 0.5574 | 0.0293 | [0.5364, 0.5784] |
| B5_replay | 10 | 0.4295 | 0.0295 | [0.4084, 0.4506] |
| V1_empFisher_l1 | 10 | 0.5678 | 0.0577 | [0.5266, 0.6091] |
| V1_empFisher_l10 | 10 | 0.5695 | 0.0502 | [0.5336, 0.6054] |
| V1_empFisher_l100 | 10 | 0.5680 | 0.0464 | [0.5348, 0.6012] |
| V2_perlayer_l1 | 10 | 0.6029 | 0.0360 | [0.5772, 0.6287] |
| V2_perlayer_l10 | 10 | 0.5394 | 0.0357 | [0.5138, 0.5649] |
| V2_perlayer_l100 | 10 | 0.5278 | 0.0315 | [0.5053, 0.5503] |
| V3_hybrid | 10 | 0.5056 | 0.0273 | [0.4861, 0.5252] |

_Context: pooled within-condition SD ≈ 0.0382 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3_warm` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B5_replay | 10 | -0.1279 | -0.1290 | -1.000 | -2.794 | 0.001953 | 0.01562 |
| V1_empFisher_l1 | 10 | +0.0104 | +0.0329 | +0.127 | +0.150 | 0.7695 | 1 |
| V1_empFisher_l10 | 10 | +0.0121 | +0.0296 | +0.164 | +0.209 | 0.6953 | 1 |
| V1_empFisher_l100 | 10 | +0.0106 | +0.0281 | +0.164 | +0.194 | 0.6953 | 1 |
| V2_perlayer_l1 | 10 | +0.0455 | +0.0467 | +0.927 | +1.395 | 0.005859 | 0.0293 |
| V2_perlayer_l10 | 10 | -0.0180 | -0.0123 | -0.891 | -1.012 | 0.009766 | 0.03906 |
| V2_perlayer_l100 | 10 | -0.0296 | -0.0264 | -1.000 | -1.803 | 0.001953 | 0.01562 |
| V3_hybrid | 10 | -0.0518 | -0.0460 | -1.000 | -1.129 | 0.001953 | 0.01562 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.05 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| B5_replay | 10 | -0.1279 | [-0.1545, -0.1014] | 0.9998 | 3.149e-07 | 0.9998 | no |
| V1_empFisher_l1 | 10 | +0.0104 | [-0.0297, +0.0505] | 0.01102 | 0.05168 | 0.05168 | no |
| V1_empFisher_l10 | 10 | +0.0121 | [-0.0215, +0.0457] | 0.004004 | 0.03415 | 0.03415 | **YES** |
| V1_empFisher_l100 | 10 | +0.0106 | [-0.0210, +0.0421] | 0.003265 | 0.02385 | 0.02385 | **YES** |
| V2_perlayer_l1 | 10 | +0.0455 | [+0.0266, +0.0644] | 3.381e-06 | 0.336 | 0.336 | no |
| V2_perlayer_l10 | 10 | -0.0180 | [-0.0284, -0.0077] | 0.0001535 | 3.667e-07 | 0.0001535 | **YES** |
| V2_perlayer_l100 | 10 | -0.0296 | [-0.0392, -0.0201] | 0.001769 | 4.693e-08 | 0.001769 | **YES** |
| V3_hybrid | 10 | -0.0518 | [-0.0784, -0.0252] | 0.5483 | 3.1e-05 | 0.5483 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.05, +0.05]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
