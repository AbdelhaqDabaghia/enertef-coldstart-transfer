# Revision v2 — positioning against neighbouring work

Our headline result ("a full fine-tune costs 1.7 % source error on EV; no CL
method helps") runs against a literature that reports continual learning working.
That divergence must be explained in the paper, not left for a reviewer to find.
Below: what each neighbour actually did, why our regime differs, and the one that
validates our corrected methodology.

---

## The three papers that matter

**(A) Prabowo, Chen, Xue, Sethuvenkatraman & Salim, BuildSys '23** —
*Navigating Out-of-Distribution Electricity Load Forecasting during COVID-19*
(doi 10.1145/3600100.3623726). FSNet applied to 13 building complexes in
Melbourne across COVID lockdowns. Concludes continual learning is "crucial"
during out-of-distribution periods.

**(B) arXiv:2510.00809** — *Foundation vs. Specialized Models: Evaluating
Catastrophic Forgetting in Continual Time Series Forecasting*. TimesFM-2.0,
Chronos-2 and SamFormer under continual fine-tuning on energy benchmarks.
Fine-tuning "consistently triggers forgetting"; DER mitigates it, and lets small
models match foundation models.

**(C) arXiv:2606.24955** — *Towards Continuous Power Forecasting: Practical
Continual Learning for Real-World Energy Systems in Nonstationary Time Series*.
23 months, 95 power entities. Compares online EWC, three real-replay variants,
and two pseudo-replay variants (CLeaR framework).

---

## Why (A) and (B) do not contradict us: different regimes

Both study **sequential adaptation over time within one domain** — a model
repeatedly updated on a non-stationary stream, where "the source" is simply its
own past. (A) is an abrupt temporal shock (lockdown) on the same buildings; (B)
is a chain of tasks learned in sequence.

Ours is **one-shot cross-site transfer**: a fixed, fully-trained source model
moved to a different physical site, with a single adaptation step, and the source
domain retained as a distinct evaluation target. There is no task sequence and no
repeated drift.

That distinction predicts the difference in findings. Forgetting accumulates over
a sequence of updates; a single fine-tune on 30 days has far less opportunity to
overwrite. Our EV result (+1.7 %) is a *lower bound* on forgetting in this family
of problems, not a refutation of (A) or (B).

**Sentence for the paper:** *"Prior reports that continual learning is essential
for energy forecasting concern sequential adaptation within one domain, where
forgetting accumulates across many updates. We study single-step cross-site
transfer, where a fully-trained source model is adapted once; we find forgetting
in this regime is an order of magnitude smaller, and correspondingly harder for
any CL method to improve upon."*

---

## Why (C) matters most — it validates the correction we just made

(C) reports **Baseline_L (the warm-up model only)** and a **Forgetting Ratio**,
defined as the relative increase in error on the warm-up dataset after
adaptation. That is precisely the pre-adaptation reference our tables were
missing, and the metric our revision now adopts. We should cite (C) as prior art
for the control rather than present it as our own innovation.

More importantly, (C) concludes there is **no universal winner**: replay wins on
reconstruction stability, while regularisation-based methods outperform replay
for forecasting generalisation under concept drift.

That is the same shape as our EV/PV split, and it is the natural place to insert
our contribution. (C) observes context-dependence; we can say **what the context
is**:

| | EV (LU→UK) | PV (LU→Konstanz) |
|---|---:|---:|
| zero-shot target nRMSE (domain gap) | 0.80 | 66.5 |
| forgetting from a full fine-tune | **+1.7 %** | **+183 %** |
| replay's effect on retention | −21.6 % (re-learning) | **+17 %** (real protection) |

**Sentence for the paper:** *"Recent work reports context-dependent results, with
no single continual-learning strategy dominating. Our two tasks suggest the
governing variable is the magnitude of the domain gap: under a moderate shift the
adaptation barely perturbs the source and no method can demonstrate a benefit,
whereas under an extreme shift forgetting is severe and replay is decisively
effective. Zero-shot transfer error is a cheap a-priori proxy for which regime a
given transfer falls into."*

---

## Claims to soften

1. Do **not** write that CL methods do not work for energy forecasting. Write
   that they do not help *in this transfer regime*, and say which regime that is.
2. Do **not** present the pre-adaptation reference as novel — cite (C).
3. Our EV finding that EWC *degrades* retention (+3.6 % to +10 % vs +1.7 % for a
   plain fine-tune) does appear to be new: (B) and (C) report regularisation
   helping or being competitive. This is worth stating explicitly as a
   disagreement, with the caveat that our regime differs.

---

## Search caveat

This positioning rests on two web searches, not a systematic review. Before
submission, run a proper search on: cross-domain transfer forecasting +
catastrophic forgetting; "forgetting ratio" / "backward transfer" in regression;
scale/representation normalisation as an alternative to CL. In particular check
whether anyone has already reported that forgetting magnitude scales with domain
distance — that is our sharpest claim and the one most likely to have a precedent.
