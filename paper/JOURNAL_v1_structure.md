# Journal paper — structure, framing and evidence map

Merges Paper A (cross-regime transfer + representation ablation) and Paper B
(autonomous CL framework + MPC, deployed site). Built only on measurements that
exist in `Data/results/`. No new method is claimed.

---

## 1. The gap

Not "nobody studied CL for energy forecasting" — many have, and they disagree.
The gap is **why they disagree**, and it is methodological:

> Continual-learning studies in energy forecasting report retention scores of
> competing methods, but almost never report what the model scored on the source
> **before any adaptation**. Without that control, one cannot tell whether a
> method preserved knowledge, or simply re-learned it — nor whether there was any
> forgetting to prevent in the first place.

Consequence, measured here: with the reference in place, a full fine-tune costs
**+1.7 %** source error on EV — there is almost nothing for a CL method to fix —
while replay scores **21.6 % better than the starting point**, i.e. it re-learns
rather than retains. Both readings are invisible without the control.

One recent work does report it (`arXiv:2606.24955`, `Baseline_L` and a Forgetting
Ratio). We adopt and generalise that practice; we do not claim it as ours.

## 2. Research question

> **Under what conditions does catastrophic forgetting actually occur in
> operational energy forecasting, and which interventions are worth their cost?**

Sub-questions, each answered by an experiment we already have:
1. How much forgetting does a *single* adaptation cause? (Paper A, EV/PV, n=10)
2. How much does it accumulate over *repeated* adaptations? (Paper B, 12 cycles × 5 seeds)
3. Does the choice of representation change the answer? (representation ablation + `cl_revin`)
4. Do regularisation-based methods help, hurt, or do nothing? (EWC bench, CL suite, MAS in Paper B)
5. Does any of it reach the operational decision? (MPC on real ENTSO-E prices)

## 3. Positioning (state of the art)

Three blocks, each with a stated limitation we address:

**(a) CL for time-series/energy forecasting.** Prabowo et al., BuildSys '23
(doi 10.1145/3600100.3623726) — FSNet across COVID lockdowns, concludes CL is
crucial. `arXiv:2510.00809` — foundation models forget under continual
fine-tuning. `arXiv:2511.17936` — replay halves forgetting on heterogeneous
streams but ties on benign ones. *Limitation:* all study sequential adaptation
within one domain; none reports the pre-adaptation control across regimes.

**(b) Regularisation-based CL in energy.** EWC for building load
(ScienceDirect S0378778822002699) reports improved stability;
`arXiv:2606.24955` finds regularisation can beat replay under drift.
*Limitation:* contradictory verdicts, unexplained. We show the verdict depends on
the regime, and quantify which.

**(c) Normalisation / representation for transfer.** RevIN-style per-instance
normalisation, domain-native reformulations (per-unit, clear-sky index).
*Limitation:* treated as preprocessing, never as a variable that changes the
CL conclusion. Our ablation shows an 8× accuracy swing (0.95 → 0.116), an order
of magnitude larger than any difference between CL algorithms.

> **Do not write** "nobody has done this". Write: existing work does not jointly
> report the pre-adaptation reference, both the single-shot and sequential
> regimes, and the representation axis — which is why its verdicts conflict.

## 4. Contributions (three, not ten)

**C1 — A measurement protocol that makes forgetting claims interpretable.**
Pre-adaptation reference + scale-invariant correlation alongside nRMSE, showing
that a large share of reported "retention" differences is amplitude
re-calibration (EV: bias −15.2 kW at corr 0.958; a pure ×1.31 rescale takes
0.5479 → 0.3873).

**C2 — A characterisation of when forgetting matters.** Single adaptation:
+1.7 % (EV). Twelve sequential adaptations: 11.1 kW backward transfer, p=0.002.
Extreme scale gap (PV, zero-shot 66.5): the model is not "forgetting" but
mis-scaled. Forgetting is a function of the number of adaptations and the size of
the shift — not an inherent property of fine-tuning.

**C3 — Operational validation.** The full chain from nightly cloud adaptation to
a convex MPC scheduler on real ENTSO-E day-ahead prices at a commercial site,
with a tolerance-gated promotion mechanism — including the engineering failure
modes (ONNX export, data-quality degradation, provenance) absent from the CL
literature.

## 5. Abstract (draft)

> Continual-learning methods are increasingly proposed to keep deployed energy
> forecasters accurate under drift, yet the literature disagrees on whether they
> help. We show that a large part of this disagreement is measurement: retention
> is reported relative to competing methods, but rarely relative to the model's
> own performance before adaptation. Introducing that control on two real
> cross-site transfers (Luxembourg → UK Electric Nation EV charging; Luxembourg →
> Konstanz PV) with ten seeds, Holm-corrected paired tests and pre-registered
> equivalence margins, we find a full fine-tune degrades source accuracy by only
> 1.7 %, while experience replay scores 21.6 % *better* than the starting point —
> it re-learns the source rather than retaining it. Regularisation-based methods
> (EWC variants, MAS, LwF) do not merely fail to help: measured against the same
> reference they degrade retention by 3.6–10 %. In a twelve-cycle sequential
> deployment at a commercial site, however, forgetting does accumulate and
> bounded replay resists it (11.1 kW backward transfer, p=0.002), whereas an
> output-sensitivity regulariser does not. A controlled representation ablation
> shows the accuracy swing attributable to representation (8×) exceeds any
> difference between continual-learning algorithms by an order of magnitude.
> Finally, we close the loop to operation: the continually updated forecasts
> drive a convex MPC scheduler under real ENTSO-E day-ahead prices. We conclude
> that forgetting in operational energy forecasting is governed by how often and
> how far the model is adapted, that representation is the first-order lever, and
> that reporting a pre-adaptation reference is a prerequisite for any retention
> claim.

## 6. Results, in the order they should appear

| # | Claim | Evidence | File |
|---|---|---|---|
| R1 | Pre-adaptation reference | EV 0.5479 (bias −15.2, corr 0.958); PV 0.6126 | `pre_adaptation_reference.csv` |
| R2 | Single adaptation barely forgets | B3 +1.7 % vs reference | `rq2_rq3_source.csv` |
| R3 | Replay re-learns, not retains | B5 = 0.4295, **below** reference | same |
| R4 | EWC degrades retention | +3.6 % to +10.0 % vs +1.7 % | `ewc_bench.csv` |
| R5 | No CL method beats warm-start on target | all p_Holm ≤ 0.01 | `cl_methods_bench.csv` |
| R6 | B3 ≡ B4 (Fisher origin irrelevant) | TOST equivalent, δ=0.02 | `stats/rq2_target.md` |
| R7 | Sequential forgetting is real | 11.1 kW BWT, p=0.002, 12 cycles × 5 seeds | Paper B |
| R8 | Representation dominates | 0.95 → 0.116 (8×) | `representation_ablation_ev.csv` |
| R9 | Persistence baseline | naive-1 0.3245; RevIN 0.1157 beats it 2.8×, MinMax 0.5130 does not | `persistence_baseline.csv` |
| R10 | Operational chain | PV MAE −21.8 %, EV MAE −67.9 % vs static; MPC on real prices | Paper B |

**Limitations to declare** (do not omit): the CL benchmark runs in the MinMax
representation, the worst ablated; on PV no representation beats naive
persistence (clear-sky 0.409 vs 0.350); one domain pair per task; the PV RQ1
regime (nRMSE 10–348) is not interpretable and is reported separately.

## 7. References to add

Verified in this session: BuildSys '23 doi 10.1145/3600100.3623726;
arXiv:2510.00809; arXiv:2606.24955; arXiv:2511.17936; arXiv:2510.21491;
ScienceDirect S0378778822002699; "Continual Learning for Time Series
Forecasting: A First Survey".

Still to add from the existing bibliographies of Papers A and B: EWC, MAS, LwF,
DER++, A-GEM, RevIN, and the MPC/ENTSO-E references. **Check before submission**
whether anyone has already reported that forgetting magnitude scales with the
number of adaptations or with domain distance — that is C2, and it is the claim
most likely to have a precedent.
