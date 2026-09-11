# Journal merge — analysis A→P

Paper 1 = *An Autonomous Continual-Learning Framework for EV Charging* (main.tex, 1935 lines).
Paper 2 = *Representation Beats Regularization in Energy Transfer* (1107 lines).
Plus the September 2026 measurements in `Data/results/`, flagged **[new]** throughout —
these are NOT in either ZIP and must be labelled as new work.

---

## A. What Paper 1 actually contributes

**Problem.** Deployed energy forecasters degrade under concept drift. Full retraining
is expensive, rolling-window retraining discards seasonal knowledge, naive fine-tuning
forgets.

**Real contribution.** An *operational architecture*, explicitly not a new CL algorithm.
Three-loop design at a commercial Luxembourg site: nightly cloud adaptation, a
tolerance-gated promotion (deploy only if no regression beyond 5 % on a recent holdout)
with fail-safe retention, and a convex MPC scheduler driven by the adapted forecasts
under real ENTSO-E day-ahead prices.

**Method.** PV LSTM and EV CNN-LSTM adapted through an EWC-inspired proximal penalty
whose importance estimator is a MAS-style output-sensitivity proxy (OSR), honestly
labelled as *not* an empirical Fisher.

**Central experiment.** 12-cycle stream (winter 2024 → winter 2025, +30-day steps,
90-day windows), 5 seeds, 4 strategies, 240 SageMaker jobs. Retention = backward
transfer on a fixed early-winter reference window.

| strategy | BWT_mid (kW) | BWT_peak | plasticity MAE |
|---|---:|---:|---:|
| naive (λ=0) | +7.04 ± 7.18 | +13.07 | 8.54 |
| **OSR (λ=1000)** | **+9.68 ± 6.37** | +15.63 | 9.37 |
| Replay | **−1.39 ± 7.70** | 6.83 | 13.82 |
| Replay+OSR | −1.77 ± 6.64 | 5.67 | 13.04 |

Replay vs OSR: −11.07 kW, same sign on 5/5 seeds, paired t p=0.0023.
λ sweep {0,10,10²,10³,10⁴}: BWT_mid stays +8 to +12 kW — **no λ restores retention**.
DER++ variant intermediate and noisy (+5.6 ± 9.6).

**The finding that matters for the merge.** Paper 1 already concludes that parameter
anchoring does not work: *"coverage of the past input distribution, rather than
parameter anchoring, [is] the operative mechanism"*. Its parameter-space diagnostic
shows cumulative weight drift ‖θ_k − θ_0‖ grows at the same rate across all four
strategies while retention diverges sharply.

**Limits.** Single site, seasonal shift only, n=5 seeds, retention measured on one
reference window, OSR importance estimated on the same rolling 90-day window (so it
protects *recent* parameters — the paper says so itself).

## B. What Paper 2 actually contributes

**Problem.** Cross-site cold-start transfer: can a data-scarce target bootstrap from a
mature production model, and keep adapting without forgetting the source?

**Real contribution.** A controlled ablation showing the *output representation*, not
the CL algorithm, governs transfer. Seven representations on EV: MinMax 0.95 → RevIN
0.116 warm-start nRMSE, an **8× swing**, against differences of a few percent between
EWC / replay / MAS / LwF / DER++ / A-GEM. Mechanistic analysis: cos(F, |g_target|) =
0.82, ~62 % of target-gradient energy in the top-10 % Fisher parameters — the Fisher
protects exactly what the target needs to move.

**Method.** Six conditions B0–B5 plus the full CL taxonomy, two real published targets
(UK Electric Nation EV, Konstanz PV), 10 seeds, Holm, dz, pre-registered TOST margins.

**Limits.** One-shot adaptation only (no sequential stream), one domain pair per task,
and the whole benchmark runs in the *worst* representation (MinMax production scalers).

## B-bis. The September measurements **[new]**

Not in either ZIP. In `Data/results/`, committed 04847bd.

1. **Pre-adaptation reference** (`pre_adaptation_reference.csv`). EV source retention
   before any adaptation = 0.5479; PV = 0.6126. Zero-shot target: EV 0.7964, PV 66.48.
2. **A single fine-tune barely forgets** in one-shot transfer: +1.7 % on EV.
3. **Replay re-learns rather than retains**: B5 = 0.4295, i.e. 21.6 % *better* than the
   starting point — it trains on 60 days of source data.
4. **Calibration contaminates the retention metric** (`retention_anomaly.py`): the
   production model under-predicts the source holdout by 15.2 kW at corr 0.958; a pure
   ×1.31 rescale takes 0.5479 → 0.3873.
5. **Representation reverses the ordering** (`cl_revin.csv`, 130 runs, n=10). Under
   RevIN all eight regularisation variants beat replay on retention (dz to −7.75,
   rank-biserial −1.000, Holm p=0.023), where under MinMax EWC degraded it.
6. **PV does not reverse** (clear-sky, smoke test): regularisation degrades +15 to +20 %.
   Full 10-seed run in progress.
7. **Source quality does not explain it** (`_srcq_smoke.csv`): at source quality 0.83 and
   0.21, regularisers help equally (MAS −0.040 then −0.039).
8. **Target leakage**: `roll_1h_mean[t] = mean(ev[t-3..t])` is trailing *and inclusive*,
   and NEXT_EXO passes the predicted step's features — the model receives a quantity
   containing ev[t] while predicting ev[t]. Not computable at inference.
9. **Persistence** (`persistence_baseline.csv`): naive-1 = 0.3245 (EV) / 0.3500 (PV).
   MinMax models lose to it; RevIN beats it 2.8× on EV; **no PV representation beats it**.
10. **Operational value bound** (`mpc_forecast_value_realprice.csv`): under real
    day-ahead prices, perfect-vs-persistence PV forecasting is worth **1.74 EUR/day
    (1.55 % of the bill)** — the ceiling for any forecast improvement on this site.

## C. What must be cut

From Paper 1: the long technology descriptions (cloud stack, framework versions), the
Bayesian derivation of EWC beyond one paragraph (it is textbook and the deployed
estimator is not Fisher anyway), the single-cycle accuracy figures (21.83 % / 67.87 %)
demoted to one sentence — they measure plasticity on the recent window and cannot speak
to forgetting, as the paper itself states.

From Paper 2: multicountry (10 countries), multitarget PV (6 sites), the LR/epoch sweep,
the domain-distance correlation study, RQ3 Fisher-scale ablation. All are a *second
paper's* worth of material and dilute the story. Keep only the representation ablation
and the mechanistic Fisher/gradient analysis.

Cut entirely: layer freezing (an artefact — the effect vanishes under RevIN), CARE
(never implemented, refuted analytically), the contention profile (a diagnostic tool).

## D. What must be kept

Paper 1's spine: three-loop architecture, OSR formulation, tolerance-gated promotion,
fail-safe retention, MPC coupling, the 12-cycle × 5-seed study, the λ sweep, the
parameter-drift diagnostic, the engineering-challenges section (rare and cited).

Paper 2's usable 30 %: the representation ablation (7 representations, 8× swing), the
Fisher/target-gradient mechanistic analysis (cos = 0.82, 62 % top-decile), and the
cross-site transfer setting as a *second shift type* (spatial, vs Paper 1's seasonal).

New: items 1–5 and 8–10 above.

## E. The new research gap

Paper 1 concludes parameter anchoring fails and data coverage governs retention. That
conclusion is drawn in one representation, on one task. **[new]** measurements show the
ordering inverts under RevIN: anchoring beats rehearsal there. Neither paper can say
which regime it is in, and neither offers a way to find out before committing to a
mechanism.

> Autonomous continual-learning pipelines for energy forecasting must commit to one
> adaptation mechanism at design time, yet the relative effectiveness of parameter
> anchoring versus rehearsal is not a property of the method: it is contingent on the
> representation the model operates in. Existing work — including our own — reports one
> ordering per study and generalises from it. What is missing is (i) a measurement
> protocol that makes retention claims comparable at all, and (ii) an autonomous
> mechanism that selects the adaptation strategy from evidence rather than assumption.

## F. Problem statement

How should an autonomously adapting energy forecaster decide *how* to adapt, when the
best adaptation mechanism depends on conditions the system cannot know in advance?

## G. Research question

> Can an autonomous continual-learning pipeline select its adaptation mechanism from
> online evidence rather than fixed design choice, and does the resulting system retain
> more of the past regime without sacrificing plasticity or downstream control quality?

## H. Hypotheses

- **H1** — The relative ordering of parameter anchoring and rehearsal is not invariant:
  it depends on the representation in which adaptation occurs. *Supported:* MinMax vs
  RevIN inversion, n=10, unanimous. *Contested:* PV clear-sky does not invert; source
  quality does not explain it (item 7).
- **H2** — Retention differences reported in the literature partly measure output
  calibration rather than knowledge. *Supported:* item 4.
- **H3** — Because no single mechanism dominates across regimes, an autonomous system
  that *validates candidates from several families each cycle* retains better than one
  committed to a single family. **Requires the champion–challenger experiment (M1).**

## I. New scientific contribution (four)

1. **A retention measurement protocol.** Pre-adaptation reference plus a scale-invariant
   correlation term, showing that a substantial share of reported retention differences
   is amplitude re-calibration. Prior art for the control: arXiv:2606.24955 (Baseline_L,
   Forgetting Ratio) — cite it, do not claim it.
2. **Evidence that the anchoring-vs-rehearsal ordering is regime-contingent**, with the
   representation ablation as the manipulated factor and three hypotheses tested and
   rejected as explanations (source quality, freezing depth, gradient conflict).
3. **Champion–challenger adaptation**: extending the existing tolerance-gated promotion
   from *one candidate* to *one candidate per mechanism family*, letting the validation
   gate that already exists select the mechanism. Justified by 1 and 2, not invented.
4. **End-to-end validation** through the MPC scheduler, with the marginal value of
   forecasting quantified rather than asserted (1.74 EUR/day ceiling).

## J. Methodology (Paper 1's pipeline, one stage extended)

Input → Monitoring → **Decision** → Adaptation → Validation → Promotion → Rollback → Deployment

Paper 1 fixes Adaptation to OSR. The extension: run *K* candidates per cycle, one per
family (naive, OSR, replay, replay+OSR), evaluate each on the same recent holdout **and
on the retained reference window**, and let the existing promotion gate choose. The
architecture already contains the selection machinery — the paper never used it to
choose *between mechanisms*, only to accept or reject one.

Cost is linear in K and small in absolute terms (models are small; Paper 1 already ran
240 jobs). This is an argument the deployment section can make quantitatively.

## K. Equations to keep or add

Keep: the OSR objective, the MAS-style importance estimator, the promotion criterion,
the convex MPC formulation. All are already in Paper 1 and all do work.

Add three, and only three:

1. **Backward transfer with an explicit reference**
   BWT(k) = MAE_ref(k) − MAE_ref(0) — already in Paper 1; make MAE_ref(0) a reported
   quantity everywhere, not an implicit zero.
2. **Calibration-corrected retention**: report ρ(ŷ, y) alongside MAE, and the residual
   after the best affine rescale a·ŷ+b, separating shape loss from scale drift.
3. **Champion–challenger selection**: m* = argmin over families of a validation score
   combining recent-window error and reference-window error, under the existing
   tolerance constraint.

Drop the full Bayesian derivation of EWC to one paragraph plus a citation.

## L. Reusable existing experiments

12-cycle × 5-seed study (240 jobs); λ sweep; DER++ variant; parameter-drift diagnostic;
deployment reliability; MPC operation; engineering challenges. From Paper 2: the
representation ablation and the Fisher/gradient mechanistic analysis. **[new]**: items
1–10.

## M. Additional experiments needed (four, ranked)

**M1 — Champion–challenger on the 12-cycle stream. *Required.*** Without it,
contribution 3 is a proposal, not a result. Re-use the existing harness: at each cycle
train all four candidates (already done — the 240 jobs exist), then *simulate* the
selection rule offline on those results and compare the resulting chain against each
fixed-mechanism chain. **This may need no new GPU time at all** if per-cycle candidate
models were retained; otherwise one re-run.

**M2 — Representation arm on the sequential stream. *Required for H1 in Paper 1's own
setting.*** The reversal is currently shown in one-shot transfer only. Run the 12-cycle
stream with the PV model in the winning representation. Without it, H1 rests on a
different experimental regime than the paper's core.

**M3 — Calibration control.** Two arms, same representation and data, differing only in
whether the source model is the deployed one or retrained with the same recipe. Script
ready (`driver_calibration_test.py`). Settles whether the MinMax result is about
representation or about the deployed model's −15.2 kW bias.

**M4 — Leakage sensitivity.** Retrain with `shift(1)` before the rolling means and check
whether conclusions move. If they do not, document the leak; if they do, everything must
be re-run. Cheap, and it decides how much re-running is required.

Not needed: PV multicountry, buffer-size sweeps, further λ exploration.

## N. Novelty audit

| claim | status |
|---|---|
| Autonomous CL pipeline for energy | **not novel alone** — Paper 1 says so itself |
| Pre-adaptation reference | **not novel** — arXiv:2606.24955 reports it. Novel: showing how much of reported retention it explains away |
| Ordering depends on representation | **plausibly novel**, no precedent found in two searches. **Verify properly** |
| Champion–challenger for CL mechanism selection | **novel in this application**; the pattern itself is standard MLOps. Frame as such |
| OSR / MAS-style regulariser | not novel, and Paper 1 does not claim it |

**Must verify before submission:** whether anyone has reported that anchoring-vs-rehearsal
ordering flips with representation. That is the paper's sharpest claim.

## O. Rejection risks

1. **"Your models lose to naive persistence."** True in the deployed representation
   (0.513 vs 0.324 EV) and true on PV in *every* representation (clear-sky 0.409 vs
   0.350). Mitigation: report it, frame the paper on adaptation dynamics rather than
   absolute accuracy, and lead with the RevIN result (0.116, 2.8× better) as evidence
   for the representation thesis.
2. **"The reversal is one task."** PV does not reverse. Mitigation: M2, and honest
   framing as regime-contingent rather than universal.
3. **"Where is the algorithmic contribution?"** Champion–challenger, and only if M1 is
   run. Otherwise the paper is a measurement-protocol paper — defensible, but say so.
4. **"n=5 seeds."** Paper 1's paired design carries it, but state the limitation.
5. **"Target leakage."** A reviewer reading `features.py` will find it. Address it
   first, via M4.
6. **"Self-contradiction with your own conference paper."** Declare the extension,
   cite the conference version, and present the representation finding as the reason
   the earlier conclusion was regime-specific.

## P. Final structure (Paper 1's skeleton, two sections changed)

1. Introduction
2. Related Work *(+ representation/normalisation as a third block)*
3. Research Gap and Problem Formulation *(rewritten around E)*
4. System Architecture *(condensed ~30 %)*
5. Continual Learning Framework *(OSR kept; Bayesian derivation cut to one paragraph)*
6. **Representation-Contingent Adaptation** *(new — Paper 2's 30 % + [new] items 1–7)*
7. **Autonomous Deployment with Champion–Challenger Selection** *(extends Paper 1 §5)*
8. MPC Integration *(condensed; + the 1.74 EUR/day bound)*
9. Experimental Setup
10. Results *(12-cycle study; representation arm; calibration control; ablation)*
11. Discussion *(the why: coverage vs anchoring, and when each wins)*
12. Limitations *(persistence, single site, leakage, n=5)*
13. Conclusion

Sections 6 and 7 are the new contribution. Everything else is Paper 1, tightened.

**Target length:** 12–14 pages. Paper 1 alone is already ~1935 LaTeX lines; the cuts in
C are what make room.
