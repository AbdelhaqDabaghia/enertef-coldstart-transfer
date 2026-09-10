# Multi-seed statistics — metric `target_nrmse`, ref `B3_warm`
_source: Data/results/ewc_bench.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3_warm** (ref) | 10 | 0.5133 | 0.0182 | [0.5003, 0.5264] |
| B5_replay | 10 | 0.6971 | 0.0300 | [0.6756, 0.7185] |
| V1_empFisher_l1 | 10 | 0.7854 | 0.0381 | [0.7582, 0.8127] |
| V1_empFisher_l10 | 10 | 0.7915 | 0.0322 | [0.7685, 0.8146] |
| V1_empFisher_l100 | 10 | 0.7937 | 0.0305 | [0.7719, 0.8156] |
| V2_perlayer_l1 | 10 | 0.7804 | 0.0300 | [0.7589, 0.8018] |
| V2_perlayer_l10 | 10 | 0.7621 | 0.0324 | [0.7389, 0.7853] |
| V2_perlayer_l100 | 10 | 0.7677 | 0.0296 | [0.7465, 0.7889] |
| V3_hybrid | 10 | 0.7256 | 0.0125 | [0.7167, 0.7345] |

_Context: pooled within-condition SD ≈ 0.0282 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3_warm` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B5_replay | 10 | +0.1837 | +0.1880 | +1.000 | +5.762 | 0.001953 | 0.01562 |
| V1_empFisher_l1 | 10 | +0.2721 | +0.2757 | +1.000 | +7.107 | 0.001953 | 0.01562 |
| V1_empFisher_l10 | 10 | +0.2782 | +0.2720 | +1.000 | +8.785 | 0.001953 | 0.01562 |
| V1_empFisher_l100 | 10 | +0.2804 | +0.2804 | +1.000 | +9.788 | 0.001953 | 0.01562 |
| V2_perlayer_l1 | 10 | +0.2670 | +0.2620 | +1.000 | +8.838 | 0.001953 | 0.01562 |
| V2_perlayer_l10 | 10 | +0.2488 | +0.2556 | +1.000 | +7.995 | 0.001953 | 0.01562 |
| V2_perlayer_l100 | 10 | +0.2543 | +0.2615 | +1.000 | +9.559 | 0.001953 | 0.01562 |
| V3_hybrid | 10 | +0.2122 | +0.2164 | +1.000 | +8.636 | 0.001953 | 0.01562 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| B5_replay | 10 | +0.1837 | [+0.1653, +0.2022] | 4.15e-09 | 1 | 1 | no |
| V1_empFisher_l1 | 10 | +0.2721 | [+0.2499, +0.2943] | 8.631e-10 | 1 | 1 | no |
| V1_empFisher_l10 | 10 | +0.2782 | [+0.2598, +0.2966] | 1.327e-10 | 1 | 1 | no |
| V1_empFisher_l100 | 10 | +0.2804 | [+0.2638, +0.2970] | 5.08e-11 | 1 | 1 | no |
| V2_perlayer_l1 | 10 | +0.2670 | [+0.2495, +0.2846] | 1.227e-10 | 1 | 1 | no |
| V2_perlayer_l10 | 10 | +0.2488 | [+0.2308, +0.2668] | 2.865e-10 | 1 | 1 | no |
| V2_perlayer_l100 | 10 | +0.2543 | [+0.2389, +0.2698] | 5.908e-11 | 1 | 1 | no |
| V3_hybrid | 10 | +0.2122 | [+0.1980, +0.2265] | 1.286e-10 | 1 | 1 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
