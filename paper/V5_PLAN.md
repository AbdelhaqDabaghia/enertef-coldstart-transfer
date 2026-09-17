# v5 — the reframe, section by section

**Decisions taken (2026-09-17):** full reframe; the SAA / value-decomposition /
RL material goes to a **second paper**; venue chosen after drafting; write now
but hold submission until a pre-specified number of settled days exists.

---

## The thesis

v4 asserts *"we deploy continual learning with an EWC/MAS regulariser"* and
then Section VIII withdraws it (p = 0.6953 raw, p = 0.4316 gauged). A paper
cannot argue against itself.

v5 asserts instead:

> Evaluation practice for forecast-driven energy management cannot detect total
> failure. We deploy, instrument and settle a real system, and show that
> reported savings decouple from delivered savings through several independent
> mechanisms — including a feature leak that reversed our own published
> conclusion.

Continual learning stops being the claim and becomes the **evidence**: it is
the thing being evaluated when the evaluation turns out to be broken.

**What paper 2 gets:** the ledger linearity, the concavity derivation, the SAA
stochastic program (+10.75 EUR/day, 118/120 days, p < 0.0001), the RL
comparison (0/120), the full value decomposition. Paper 1 *diagnoses* the
truncation defect and leaves it open; paper 2 solves it. The sequencing is
honest and each paper has one thesis.

---

## Migration

| v4 | v5 | action |
|---|---|---|
| Title: *An Autonomous Continual-Learning Framework for EV Charging* | new title | asserts a framework; the paper's own results retract its distinguishing mechanism |
| Abstract (~650 words) | ~250 words | far over length; buries the findings behind the architecture |
| I Introduction, 5 contributions (4 contradicted) | I, 5 rewritten | see below |
| II Related Work — ML forecasting, CL, research gap | II | **pivot**: predict-then-optimise, evaluation validity, deployment reporting. The gap becomes *deployment papers report planned savings, never settled ones* |
| III Site description, three-loop architecture | III | **keep**, condense |
| III-C Dataset and data layer | III-C | **keep** |
| IV Continual Learning Framework (problem, notation, architectures) | IV | **keep**, condensed — it is the system under test |
| IV-F Output-sensitivity regularisation, ll. 626–871 (~245 lines: Bayesian motivation, Fisher, importance, gradient dynamics, limiting behaviour) | one paragraph + citation | 12 % of the paper deriving a mechanism the results show does nothing. Keeping it signals a methodological contribution and invites the reviewer onto our weakest ground |
| V Validation-gated promotion, presented as a strength | **VI**, presented as a **measured defect** | the deployed gate promotes a model naive persistence beats by 2.3× |
| V-B Statistical basis | **keep** — moves to the methods of VI | |
| VI MPC formulation | V | **keep verbatim**; it defines the decision the forecasts feed |
| VI-B The forecast-to-decision gap | VII | becomes the decoupling result |
| VII Experimental Setup | VIII | keep |
| VIII Results | IX | reorganised around the thesis |
| IX Engineering Challenges | **dissolved** | becomes the defect table in VI; a reviewer skims a section called "challenges" |
| X Discussion / limitations / generalisability | X | rewritten: what is site-specific, what is architectural |
| XI Conclusion | XI | rewritten |
| — | **NEW IV-G / VI-A: settlement methodology** | planned / counterfactual / settled, and the actuation-record boundary |
| — | **NEW VI: the defect table** | one row per mechanism, with *how it was found* as a column |
| — | **NEW IX-x: the guard suite** | each defect class mapped to an executable test |

---

## The five contributions, rewritten

Each is traceable to a file under `Data/results/` or to a commit.

1. **A settlement methodology for forecast-driven site control**, distinguishing
   *planned*, *counterfactual* and *settled* savings, together with the
   actuation-record boundary that says when the database may be trusted about
   what the controller did. Deployed and instrumented, not proposed.

2. **A measured account of silent no-op degradation**: components that answer
   plausibly instead of refusing. Seven independent instances at one site — a
   price source falling back to a constant, a promotion gate comparing only
   relatively, an ingest logging `0 rows upserted` as `[OK]`, a lookback filled
   with medians, a KPI row filtered on a name that no longer exists, a forecast
   split by charger capacity where consumption splits the other way, and a plan
   truncated by the physical floor. The damning column is *how it was found*:
   mostly a human noticing a display looked wrong.

3. **Quantified train/serve feature skew on a live system.** The same weights
   on the same holdout, re-scored under the convention actually served: nRMSE
   0.548 → 1.254, correlation with ground truth 0.958 → 0.458. A causal refit
   recovers most of it (nRMSE 0.8495, bias −4.2 kW). Controlled ablations of
   this kind are almost absent from the literature.

4. **A falsification of our own published contribution.** Under the leaky
   pipeline the regulariser beat fine-tuning; under causal features it is
   indistinguishable (p = 0.6953 raw, p = 0.4316 gauged). What survives is
   stronger: bounded rehearsal beats fine-tuning by 0.346 nRMSE, unanimously
   over ten seeds (d_z = −17.57, p = 0.0020), and is the only mechanism that
   beats naive persistence.

5. **Evidence that forecast accuracy does not predict decision value**, and a
   bound on what arbitration could recover: rehearsal forecasts 34 % better on
   35/35 day–seed pairs while realised controller cost is unchanged
   (p = 0.645), and an oracle choosing daily with knowledge of the outcome
   beats the best fixed choice by 0.41 %.

Plus the artefact: a guard suite of 57 tests across 7 files, in CI, mapping
each defect class to an executable check.

---

## What must exist before submission

- **Settled days.** Zero as of 2026-09-17; the first is 2026-09-18. The
  framing promises settled euros, so *n* is pre-registered before the data
  exists (`PREREGISTRATION_prospective.md`) and submission waits for it.
- Verification of `li2024building` and `ng2024cost`.
- Overleaf compile of the merged v5.
- Co-author discussion — blocks **submission**, not drafting.

## Numbers that must not be reused

- **"Correcting the objective is worth +7.91 EUR/day."** True on 25 days,
  it collapses to +3.45 with p = 0.4976 on 120. Belongs to paper 2, and only
  in its 120-day form.
- **nRMSE 1.389** and **1.254** are different measurements — the production
  weights under the serving convention, and the causal retained-task reference.
  Label both explicitly; do not let one stand in for the other.
