# Revision v2 — the persistence baseline must be reported

`coldstart_transfer/baseline_persistence.py` exists to answer "did you beat
persistence?" and its docstring says so. The answer is currently absent from the
manuscript, and it is not uniformly favourable. A reviewer who opens the
repository finds `Data/results/persistence_baseline.csv` immediately.

All numbers below: same 14-day target holdout, same nRMSE definition
(`trainer.nrmse`), same one-step horizon. Directly comparable.

---

## EV (LU → UK): the model wins, but only in the right representation

| predictor | target nRMSE | vs naive-1 |
|---|---:|---|
| **naive-1** (last observation, 15 min ago) | **0.3245** | — |
| seasonal-week (same time last week) | 0.4857 | worse |
| seasonal-day (same time yesterday) | 0.6138 | worse |
| | | |
| B0 from-scratch, production scalers | 1.0527 | **3.2× worse** |
| B5 replay, production scalers | 0.6993 | 2.2× worse |
| B3 warm-start, production scalers | 0.5130 | **1.6× worse** |
| B3 warm-start, **RevIN** | **0.1157** | **2.8× better** |

The models the CL benchmark is built on **lose to naive persistence**. The same
architecture in the RevIN representation beats it by 2.8×.

This is the sharpest evidence available for the paper's central thesis: the
bottleneck is the representation, not the algorithm. An 8× swing in accuracy
(0.95 → 0.116 across representations) dwarfs every difference between EWC,
replay, A-GEM, DER++ and MAS. Reporting it strengthens the paper.

## PV (LU → Konstanz): no representation beats persistence

| predictor | target nRMSE | vs naive-1 |
|---|---:|---|
| **naive-1** | **0.3500** | — |
| seasonal-week | 0.6349 | worse |
| seasonal-day | 0.6614 | worse |
| | | |
| B3 warm-start, production scalers | 10.14 | 29× worse |
| B3 warm-start, RevIN | 0.6445 | 1.8× worse |
| B3 warm-start, log | 0.6611 | 1.9× worse |
| B3 warm-start, **clear-sky** (best) | **0.4090** | **1.17× worse** |

Even the best representation on PV — the clear-sky index, the paper's own
domain-informed contribution — remains **17 % worse than copying the last
observation**. This must be stated. Claiming the PV replication "obtains the same
verdict" while the best PV model loses to a one-line baseline is not defensible.

---

## Honest reading

One-step-ahead at 15 min resolution is the horizon where persistence is hardest
to beat: the best estimate of power in fifteen minutes is usually the power now.
PV at 15 min is especially autocorrelated (irradiance varies slowly except under
cloud transients), which is why persistence is strong there and why the margin
against it is thin.

That explanation is legitimate but it does not rescue the comparison, because
persistence faces exactly the same signal. Two defensible responses:

1. **Report and scope** (cheap, honest). State that at the 15-minute one-step
   horizon persistence is a strong baseline, that the EV models beat it once the
   representation is corrected, and that the PV models do not. Frame the paper's
   subject as *transfer dynamics of a deployed forecaster*, not as
   state-of-the-art absolute accuracy.
2. **Add a longer horizon** (stronger, costs a re-run). Evaluate at 1 h / 4 h /
   24 h, where persistence degrades quickly and a learned model should pull ahead.
   If the models win there, the objection disappears and the paper gains an
   operationally meaningful result. Note this requires re-training: the current
   models predict a single step, and the target/exogenous windows would have to be
   shifted by the horizon.

Not reporting it at all is the one option to exclude.

---

## Suggested text

> *We benchmark against naive and seasonal persistence on the identical holdout
> and metric. At the 15-minute one-step horizon persistence is a demanding
> reference: in the deployed MinMax representation our transferred models do not
> beat it (0.513 vs 0.324 on EV), whereas the same architecture under RevIN does,
> by 2.8× (0.116). On PV the best representation (clear-sky, 0.409) remains 17 %
> short of naive persistence (0.350). We report this because it isolates our
> central claim: the accuracy swing attributable to representation (8× on EV)
> is an order of magnitude larger than any difference between the continual-
> learning algorithms we benchmark. Our contribution concerns the dynamics of
> cross-regime transfer and retention, not absolute single-step accuracy, for
> which persistence remains competitive at this horizon.*
