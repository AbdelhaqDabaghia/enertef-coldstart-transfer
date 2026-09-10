# Revision v2 — claims that must change, and replacement text

Every number below comes from a file in `Data/results/`. Nothing here is an
estimate. Generated after adding the pre-adaptation reference that all retention
tables were missing (`scripts/pre_adaptation_reference.py`).

---

## 0. The measurement that changes the reading of every retention table

The retention tables compared conditions against each other but never against the
**starting point**: what the transferred model scores *before* any target
fine-tuning. That reference now exists:

| task | pre-adaptation source retention | bias | corr |
|---|---:|---:|---:|
| EV (LU→UK) | **0.5479** | −15.2 kW | 0.958 |
| PV (LU→Konstanz) | **0.6126** | −0.88 kW | 0.944 |

Zero-shot target error before adaptation: EV **0.7964** (corr 0.602),
PV **66.48** (corr 0.938).

Two consequences, both of which the paper currently gets wrong:

1. A condition scoring **below** the reference has not *preserved* the source —
   it has *improved* on it. Replay scores 0.4295 on EV, i.e. 21.6 % **better**
   than the model it started from. It re-trains on 60 days of source data, so
   this is re-learning, not retention.
2. `scripts/retention_anomaly.py` shows the production model under-predicts the
   source holdout by 15.2 kW while keeping corr = 0.958. A pure scale refit
   (×1.31) takes 0.5479 → 0.3873. So most of the "retention" spread on EV is
   **amplitude re-calibration**, not knowledge preservation. Any condition that
   re-trains the output head can capture part of it.

**Action:** add the pre-adaptation row to every retention table, and report
correlation alongside nRMSE so calibration error is distinguishable from loss of
signal shape. Block ready to paste: `Data/results/stats/pre_adaptation_reference.md`.

---

## 1. "EWC adds nothing" → **EWC actively harms retention**

Current abstract: *"EWC adds nothing over a plain warm-start fine-tune."*

Measured against the pre-adaptation reference (EV, 10 seeds):

| condition | retention | vs reference |
|---|---:|---|
| B5 replay | 0.4295 | −21.6 % (improves) |
| V3_hybrid | 0.5056 | −7.7 % (improves) |
| V2_perlayer_l100 | 0.5278 | −3.7 % (improves) |
| **B3 warm (plain fine-tune)** | 0.5574 | **+1.7 %** |
| V1_empFisher (all three λ) | ~0.568 | +3.6 to +3.9 % |
| V2_perlayer_l1 | 0.6029 | **+10.0 %** |

Every EWC variant except the two strongly-normalised ones ends up **worse than
doing nothing**. The claim is stronger than the paper currently makes it: EWC
degrades the very quantity it is designed to protect.

**Replacement:** *"EWC does not merely fail to help: measured against the
pre-adaptation reference, every empirical-Fisher variant degrades source
retention (+3.6 % to +10.0 %) where a plain warm-start fine-tune costs only
+1.7 %."*

---

## 2. "Replicated on PV, same verdict" → **the two tasks disagree on forgetting**

Current abstract: *"we replicate the entire protocol on a second, physically
distinct task ... and obtain the same verdict."*

The ranking of methods does replicate. **The amount of forgetting does not:**

| condition | EV (ref 0.5479) | PV (ref 0.6126) |
|---|---:|---:|
| B3 warm | **+1.7 %** | **+183 %** |
| B5 replay | −21.6 % | +17 % |
| B1 EWC-source | +12.3 % | +156 % |

On EV a full fine-tune costs 1.7 % — there is essentially **no catastrophic
forgetting to prevent**, which is why no CL method can demonstrate a benefit. On
PV the same fine-tune nearly triples source error, and replay cuts that from
+183 % to +17 %: here forgetting is real and replay genuinely prevents it.

Saying "the same verdict" is not defensible as written, and a reviewer checking
the PV retention column will see it.

**Replacement claim (this is the paper's strongest available framing):**

> *Catastrophic forgetting is not a given: its magnitude is governed by how far
> the target regime sits from the source. Under a moderate shift (EV, zero-shot
> nRMSE 0.80) a full fine-tune costs 1.7 % source error and no continual-learning
> method — regularisation or replay — provides a practically significant benefit;
> several make retention worse. Under an extreme shift (PV, zero-shot nRMSE 66.5,
> a ~50× scale gap) forgetting is severe and replay is highly effective. The two
> tasks therefore bracket the phenomenon rather than confirming one another, and
> in both cases correcting the output representation addresses the cause — the
> scale mismatch — rather than the symptom.*

This reconciles the paper with the CL-for-forecasting literature (which reports
replay working) instead of contradicting it, and it keeps the title's thesis.

---

## 3. Seeds: three → ten

Abstract says *"three seeds"*. All central tables are now n=10 (`rq1_coldstart`,
`ewc_bench`, `cl_methods_bench`, `pv_rq1_coldstart`, `pv_rq2_rq3_source`, and
`rq2_rq3_source` regenerated with `INCLUDE_RQ2=1`). Equivalence margins were
pre-declared in `Data/results/stats/PREREGISTRATION.md` **before** the campaign.

Add to the abstract: seed-matched Wilcoxon, Holm correction, paired dz, and TOST
equivalence at pre-declared δ (0.02 target / 0.05 retention).

---

## 4. Persistence baseline must be reported

`Data/results/persistence_baseline.csv`, same holdout, same metric, same horizon:

| predictor | EV target |
|---|---:|
| naive-1 (last observation) | **0.3245** |
| seasonal-week | 0.4857 |
| seasonal-day | 0.6138 |
| production-scaler models (B3) | 0.5130 |
| **RevIN representation (B3)** | **0.1157** |

In the production MinMax representation the models **lose to naive persistence**.
In RevIN they beat it by 2.8×. This must be stated explicitly: it is the sharpest
available evidence for the paper's central claim that representation, not
algorithm, is the bottleneck. Leaving it unreported invites the standard reviewer
question with no answer prepared.

---

## 5. Scope limit to state honestly

The entire CL benchmark runs on the frozen production MinMax scalers — the worst
of the seven representations ablated (B3 = 0.95 vs 0.116 for RevIN). Conclusions
about EWC/replay/A-GEM/DER++ are therefore established in the regime where the
model performs worst.

A partial check exists: a freeze-depth sweep re-run under RevIN
(`Data/results/freeze_revin.csv`, 10 seeds) shows the effect sizes of CL
interventions shrink but do not vanish — replay keeps roughly half its relative
effect (−11.1 % retention for +16.5 % target, vs −23.0 % / +36.3 % under MinMax).
Under RevIN the pre-adaptation reference is 0.2034 and a full fine-tune gives
0.2355 (+15.8 %).

State this as a limitation, or re-run `ewc_bench` and `cl_methods_bench` under
RevIN to close it.

---

## 6. Suggested abstract edits (minimal diff)

- *"three seeds"* → *"ten seeds, with Holm-corrected paired tests and
  pre-registered TOST equivalence margins"*
- *"EWC adds nothing over a plain warm-start fine-tune"* → *"EWC degrades source
  retention relative to both the pre-adaptation reference and a plain warm-start
  fine-tune"*
- *"and obtain the same verdict"* → *"and obtain the same method ranking but a
  markedly different amount of forgetting, which lets us bound when forgetting
  matters at all"*
- Add one sentence: *"Measured against the pre-adaptation reference — a control
  absent from prior cold-start transfer studies — a full fine-tune costs only
  1.7 % source error on EV, so there is little forgetting for any CL method to
  prevent."*
