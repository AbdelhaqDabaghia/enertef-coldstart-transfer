# Multi-seed statistics — metric `target_nrmse`, ref `B3_warm`
_source: Data/results/cl_revin.csv | grouping: `notes` | lower is better_

## Descriptives
| condition | n | mean | std | 95% CI |
|---|---:|---:|---:|---|
| **B3_warm** (ref) | 10 | 0.1162 | 0.0035 | [0.1137, 0.1187] |
| AGEM | 10 | 0.1262 | 0.0081 | [0.1204, 0.1320] |
| B5_replay | 10 | 0.1327 | 0.0064 | [0.1281, 0.1373] |
| DERpp | 10 | 0.1605 | 0.0102 | [0.1532, 0.1678] |
| LwF | 10 | 0.1696 | 0.0106 | [0.1620, 0.1771] |
| MAS | 10 | 0.1415 | 0.0051 | [0.1379, 0.1451] |
| V1_empFisher_l1 | 10 | 0.1199 | 0.0026 | [0.1181, 0.1218] |
| V1_empFisher_l10 | 10 | 0.1294 | 0.0036 | [0.1268, 0.1319] |
| V1_empFisher_l100 | 10 | 0.1372 | 0.0052 | [0.1335, 0.1409] |
| V2_perlayer_l1 | 10 | 0.1192 | 0.0049 | [0.1157, 0.1227] |
| V2_perlayer_l10 | 10 | 0.1265 | 0.0026 | [0.1246, 0.1284] |
| V2_perlayer_l100 | 10 | 0.1308 | 0.0025 | [0.1290, 0.1327] |
| V3_hybrid | 10 | 0.1407 | 0.0032 | [0.1384, 0.1430] |

_Context: pooled within-condition SD ≈ 0.0053 (a sanity reference for choosing δ — NOT the margin itself)._

## Paired contrast vs `B3_warm` (seed-matched)
| condition | n_pairs | Δmean | Δmedian | rank-biserial | Cohen's dz | p (Wilcoxon) | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGEM | 10 | +0.0100 | +0.0112 | +0.782 | +0.944 | 0.02734 | 0.05469 |
| B5_replay | 10 | +0.0165 | +0.0179 | +1.000 | +1.978 | 0.001953 | 0.02344 |
| DERpp | 10 | +0.0443 | +0.0427 | +1.000 | +3.978 | 0.001953 | 0.02344 |
| LwF | 10 | +0.0534 | +0.0496 | +1.000 | +5.926 | 0.001953 | 0.02344 |
| MAS | 10 | +0.0253 | +0.0259 | +1.000 | +4.238 | 0.001953 | 0.02344 |
| V1_empFisher_l1 | 10 | +0.0038 | +0.0035 | +0.964 | +1.686 | 0.003906 | 0.02344 |
| V1_empFisher_l10 | 10 | +0.0132 | +0.0133 | +1.000 | +3.926 | 0.001953 | 0.02344 |
| V1_empFisher_l100 | 10 | +0.0210 | +0.0223 | +1.000 | +3.127 | 0.001953 | 0.02344 |
| V2_perlayer_l1 | 10 | +0.0030 | +0.0030 | +0.782 | +0.903 | 0.02734 | 0.05469 |
| V2_perlayer_l10 | 10 | +0.0103 | +0.0103 | +1.000 | +3.764 | 0.001953 | 0.02344 |
| V2_perlayer_l100 | 10 | +0.0147 | +0.0149 | +1.000 | +6.127 | 0.001953 | 0.02344 |
| V3_hybrid | 10 | +0.0245 | +0.0258 | +1.000 | +4.285 | 0.001953 | 0.02344 |

_Δ = condition − ref; Δ<0 means the condition is better (lower error). Holm corrects across the whole family of contrasts shown._

## Equivalence — paired TOST at δ = ±0.02 (α = 0.05)
| condition | n_pairs | Δmean | 90% CI | p_lower | p_upper | p_TOST | equivalent to ref? |
|---|---:|---:|---|---:|---:|---:|:--:|
| AGEM | 10 | +0.0100 | [+0.0039, +0.0162] | 4.542e-06 | 0.007988 | 0.007988 | **YES** |
| B5_replay | 10 | +0.0165 | [+0.0117, +0.0214] | 1.143e-07 | 0.11 | 0.11 | no |
| DERpp | 10 | +0.0443 | [+0.0378, +0.0507] | 1.012e-08 | 1 | 1 | no |
| LwF | 10 | +0.0534 | [+0.0481, +0.0586] | 4.817e-10 | 1 | 1 | no |
| MAS | 10 | +0.0253 | [+0.0219, +0.0288] | 9.079e-10 | 0.9899 | 0.9899 | no |
| V1_empFisher_l1 | 10 | +0.0038 | [+0.0025, +0.0050] | 4.338e-11 | 1.277e-09 | 1.277e-09 | **YES** |
| V1_empFisher_l10 | 10 | +0.0132 | [+0.0112, +0.0151] | 8.547e-11 | 5.906e-05 | 5.906e-05 | **YES** |
| V1_empFisher_l100 | 10 | +0.0210 | [+0.0171, +0.0249] | 6.187e-09 | 0.6742 | 0.6742 | no |
| V2_perlayer_l1 | 10 | +0.0030 | [+0.0011, +0.0050] | 2.13e-09 | 3.085e-08 | 3.085e-08 | **YES** |
| V2_perlayer_l10 | 10 | +0.0103 | [+0.0087, +0.0119] | 3.195e-11 | 7.343e-07 | 7.343e-07 | **YES** |
| V2_perlayer_l100 | 10 | +0.0147 | [+0.0133, +0.0160] | 2.808e-12 | 2.925e-05 | 2.925e-05 | **YES** |
| V3_hybrid | 10 | +0.0245 | [+0.0212, +0.0278] | 7.236e-10 | 0.9828 | 0.9828 | no |

_Equivalent ⇔ the 90% CI of Δmean lies entirely within [−0.02, +0.02]. This — not a non-significant difference test — is what licenses 'condition X adds nothing over the reference'._
