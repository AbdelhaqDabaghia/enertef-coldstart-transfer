# EnerTEF cold-start transfer — all results, 4–9 September 2026

Every number below comes from a file in `Data/results/`. Reproducible scripts are
named. Nothing here is an estimate or a projection.

Context: Paper A = cross-regime transfer study (LU→UK EV, LU→Konstanz PV).
Paper B = autonomous CL framework + MPC, accepted at a Hong Kong conference.

---

## 1. Ten-seed campaign (the statistical base)

All central tables regenerated at **n=10** with seed-matched Wilcoxon, Holm
correction, paired Cohen's dz, and TOST equivalence at margins **pre-declared
before the campaign** (`Data/results/stats/PREREGISTRATION.md`, dated 4 Sept:
δ=0.02 target, δ=0.05 retention).

Files: `rq1_coldstart.csv`, `ewc_bench.csv`, `cl_methods_bench.csv`,
`pv_rq1_coldstart.csv`, `pv_rq2_rq3_source.csv`, `rq2_rq3_source.csv`
(the last regenerated with `INCLUDE_RQ2=1`). 7 h 33 of GPU.

Key changes from the 3-seed version: `B0` at N=30 moves from 0.92±.05 to
**1.05±.14** — the standard deviation triples. The reviewers' objection to three
seeds was well founded. `B3`/`B4` barely move (0.53 → 0.51) and remain
TOST-equivalent (δ=0.02), which strengthens RQ3.

Regenerated LaTeX tables: `paper/tables_v2.tex`.

## 2. The missing control: pre-adaptation reference

**This is the methodological core of the revision.** Retention tables compared
conditions against each other but never against the model's own score *before any
adaptation*. Script: `scripts/pre_adaptation_reference.py`.

| task | pre-adaptation retention | bias | corr |
|---|---:|---:|---:|
| EV (LU→UK) | **0.5479** | −15.2 kW | 0.958 |
| PV (LU→Konstanz) | **0.6126** | −0.88 kW | 0.944 |

Zero-shot target error: EV **0.7964** (corr 0.602), PV **66.48** (corr 0.938).

Three findings that are invisible without this control:

**(a) A single fine-tune barely forgets.** EV: `B3` = 0.5574, i.e. **+1.7 %**
above the starting point. The phenomenon the whole CL benchmark addresses is
almost absent in one-shot transfer.

**(b) Replay does not retain — it re-learns.** EV: `B5` = 0.4295, i.e. **21.6 %
BETTER** than the model it started from. It trains on 60 days of source data, so
this is additional learning, not preservation. Comparing B5's "retention" to B3's
does not measure protection against forgetting.

**(c) Most of the retention spread is amplitude re-calibration, not knowledge.**
`scripts/retention_anomaly.py`: the production model under-predicts the source
holdout by 15.2 kW while keeping corr = 0.958 — the shape is perfectly preserved,
only the scale is wrong. A pure ×1.31 rescale takes 0.5479 → **0.3873**. Any
condition that re-trains the output head captures part of that. The bias exists
*before* adaptation, on recent LU data, for a model dated 18/07/2026 — consistent
with temporal drift.

Prior art for the control: `arXiv:2606.24955` reports `Baseline_L` and a
Forgetting Ratio. **Cite it; do not claim the control as novel.**

## 3. The representation reversal (the strongest new result)

`ewc_bench` and `cl_methods_bench` both run on the frozen production MinMax
scalers — the **worst** of the seven representations ablated (B3 = 0.95 vs 0.116
for RevIN). We re-ran the entire CL taxonomy under RevIN, reusing
`trainer.run_condition` and `cl_methods.run_cl_method` **verbatim** so only the
representation differs. Script: `scripts/driver_cl_revin.py`, 130 runs, n=10.

Pre-adaptation reference under RevIN: **0.2032**.

Retention vs plain fine-tune (`B3_warm`), Holm-corrected:

| condition | Δ retention | dz | p (Holm) |
|---|---:|---:|---:|
| V1_empFisher_l10 | **−0.0435** | −6.79 | 0.023 |
| MAS | −0.0419 | **−7.75** | 0.023 |
| V1_empFisher_l100 | −0.0419 | −6.62 | 0.023 |
| V2_perlayer_l100 | −0.0384 | −6.60 | 0.023 |
| V3_hybrid | −0.0377 | −6.43 | 0.023 |
| **B5_replay** | **−0.0316** | −3.02 | 0.023 |
| AGEM | −0.0135 | −1.65 | 0.023 |
| DERpp | +0.0076 | +0.58 | 0.13 |
| LwF | +0.0431 | +5.73 | 0.023 |

All eight regularisation variants beat replay. Rank-biserial = −1.000 for every
one: unanimous across all ten seeds.

**The ordering inverts:**

| | MinMax | RevIN |
|---|---:|---:|
| plain fine-tune, vs reference | +1.7 % | **+16.5 %** |
| EWC / MAS, vs reference | +3.6 to +10 % | **−4.2 to −4.9 %** |
| replay, vs reference | −21.6 % | +0.9 % |
| best retention method | **replay** | **MAS / EWC** |

A further nuance: under RevIN there is a genuine stability–plasticity trade-off
(regularisers gain retention but lose target accuracy: MAS 0.1415 vs B3 0.1162).
Under MinMax, EWC was simply dominated — worse on both axes. **The trade-off only
exists in the good representation.**

Consequence: the central conclusions of Paper A ("EWC adds nothing") and Paper B
("the output-sensitivity regulariser does not improve retention") are artefacts
of the MinMax representation.

**Not yet replicated on PV.** Required before making this the headline claim.

## 4. Negative results that closed off blind alleys

**CARE** (conflict-aware replay + elasticity, proposed externally) was refuted on
paper before any implementation. Its penalty term is `(λ/2)Σ F̄·a·(θ−θ*)²` with
a ∈ [0,1], hence always ≥ 0: a = 0 degenerates to plain replay, so CARE is
lower-bounded by replay on target error (0.699). Its own "strong success"
criterion required ≈0.51. Structurally unreachable. Measurement confirmed the
mask would also be inert: `cos(g_source, g_target)` is **positive in 11 of 12
layers** (EV), median +0.27 — there is almost no directional conflict to mask.

**Layer freezing.** `scripts/contention_profile.py` showed per-layer contention
varies strongly (top-decile Fisher enrichment 0.26–9.04× chance on EV; the pooled
6.29× reproduces the paper's "62 % in top-10 % Fisher"). Freezing the sequence
trunk looked excellent under MinMax — dominating both B3 and B5 (target 0.5067,
retention 0.4196, 10 seeds). **It vanishes under RevIN**: the effect drops from
−24.7 % to −1.5 % relative, and the L4/L5 collapse disappears entirely. Another
MinMax artefact. Files: `freeze_sweep.csv`, `freeze_revin.csv`.

**A RevIN framing objection was tested and dismissed.** RevIN de-normalises with
observed statistics, so one might argue the network does less work. The null model
(`pred = µ`) scores **0.8455** on target and 1.3470 on source — worse than
persistence. The network does the work; the margin for forgetting is wide.

## 5. Persistence baseline — must be reported

Same holdout, same metric, same one-step horizon (`persistence_baseline.csv`).

| | EV target | PV target |
|---|---:|---:|
| naive-1 (last observation) | **0.3245** | **0.3500** |
| seasonal-week | 0.4857 | 0.6349 |
| production-scaler B3 | 0.5130 ✗ | 10.14 ✗ |
| best representation B3 | **0.1157 ✓** (RevIN) | 0.4090 ✗ (clear-sky) |

In the deployed representation the models **lose to naive persistence**. Under
RevIN, EV beats it by 2.8×. **On PV, no representation beats it** — clear-sky
remains 17 % short. This must be stated; the script's own docstring says it exists
to answer "did you beat persistence?".

## 6. Operational chain: what a better forecast is actually worth

`scripts/mpc_forecast_value.py`, closed-loop on the real site
(`ECC_master_PV_EMOB1_EMOB2_15min.csv`, verified identical to
`lux_source_features.csv`: corr 1.0, max diff 0.0 over 47,524 rows). Costs are
**realised** (plan decided on the forecast, bill settled on the real PV) — the
MPC's own objective value is computed against the forecast and is not comparable
across forecasters.

| PV forecast | flat price | real day-ahead prices |
|---|---:|---:|
| perfect (bound) | 157.37 €/day | 112.03 €/day |
| persistence | +0.39 (+0.25 %) | **+1.74 (+1.55 %)** |
| flat | +37.76 | +9.13 |

Realistic price variability multiplies the value of forecasting by 4.5×, but
**1.74 €/day (~635 €/year) is the absolute upper bound** on what any forecast
improvement can deliver on this site. An oracle would earn that; a model earns a
fraction. Frame contribution as *quantifying* forecast value, not asserting it.

Caveats: greedy stand-in scheduler (not `SiteMPC`); prices from an adjacent window
(ENTSO-E API returns 404 on all documented endpoints, two tokens, both hosts,
Windows and WSL — the service appears migrated); EV demand assumed known. The
demand-forecast arm gave +0.08 €/day but that figure is **not usable**: only the
daily total is predicted and vehicle-availability constraints are not modelled.

## 7. Two technical issues found in the pipeline

**Target leakage in the rolling means.** `features.engineer_features` defines them
trailing *and inclusive*: `roll_1h_mean[t] = mean(ev[t-3..t])`. `NEXT_EXO` passes
the predicted step's features to the model, so it receives a quantity containing
`ev[t]/4` while predicting `ev[t]` (same for the 6 h and 24 h means). At inference
time this input is not computable. **How does Service 1 build `X_next` in
production?** If via the same function, there is a train/serve mismatch, and the
reported one-step scores benefit from information unavailable in deployment.

**Pre-adaptation calibration bias.** See §2(c). Directly relevant to Reviewer 2's
question on the conference paper (when was the static baseline trained relative to
EMOB2 commissioning) — the answer can now be given with a number, not a date.

## 8. Literature positioning

- **Prabowo et al., BuildSys '23** (doi 10.1145/3600100.3623726): FSNet across
  COVID lockdowns, CL "crucial". *Sequential adaptation within one domain.*
- **arXiv:2510.00809**: foundation models forget under continual fine-tuning.
  *Chained tasks.*
- **arXiv:2606.24955**: no universal winner — replay wins on reconstruction
  stability, regularisation under concept drift. **Reports `Baseline_L` and a
  Forgetting Ratio.**
- **arXiv:2511.17936**: replay halves forgetting on heterogeneous streams, ties on
  benign ones.
- **ScienceDirect S0378778822002699**: EWC improves stability for building load.

Ours is **one-shot cross-site transfer**, not sequential adaptation — which
predicts less forgetting, and explains rather than contradicts these results.
Do **not** write "nobody has done this". Write: existing work does not jointly
report the pre-adaptation reference, both regimes, and the representation axis.

## 9. What remains open

1. Replicate the representation reversal on PV (code ready, ~4 h GPU).
2. Verify the `X_next` construction in the deployed Service 1.
3. Wire `SiteMPC` and a real overlapping price series into the forecast-value study.
4. Check whether anyone has already reported forgetting magnitude scaling with the
   number of adaptations or with domain distance.
5. Camera-ready for Hong Kong: static-model training date, ONNX workaround,
   shortened abstract, and a scope sentence limiting the retention result to the
   deployed representation.
