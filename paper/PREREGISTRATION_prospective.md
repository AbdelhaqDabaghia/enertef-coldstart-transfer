# Pre-registration — prospective evaluation of adaptation policies at the Copal site

**Status: written before any day of data has been observed. Nothing below may be
changed after the first row is written; changes go in an amendment section with
their own date, and amendments are reported in the paper.**

- Registered on: _to be filled on commit — the commit date IS the registration date_
- Registration commit: _hash of this file's first commit_
- Frozen weights commit / S3 paths: _to be filled at freeze time_
- First target day: _first `target_timestamp` written under this protocol_
- **Reading date: first target day + 90 days.** No analysis of the primary
  endpoint before that date.

---

## 1. Why prospective

Every result in this project so far is retrospective: the holdout windows, the
seeds and the candidate set were chosen after the data were visible. That is
normal practice and it is also the reason each result can be argued with.

This protocol fixes the candidates, the metric and the decision rule in advance,
and lets a live site decide. The `enertef-leneda-ingest` job already delivers
J-1 actuals daily into `historical.telemetry_15min`; the `forecasts` and
`mpc_runs` tables already hold predictions and controller runs. No new
infrastructure is required, which is precisely why this is cheap to run while
the current manuscript is being written.

**What this does NOT establish.** One site, one architecture. A longer record
adds time, not sites. The generalisation objection — that ten seeds sample
initialisation variance and not population variance — is untouched by this
protocol and must not be claimed as answered by it.

## 2. Candidates

Four adaptation policies, applied nightly to the live site from the same frozen
starting point, each writing its own forecast to `forecasts` under its own
`ai_models.id`:

| label | policy |
|---|---|
| `frozen` | no adaptation; the deployed model as of the freeze date |
| `finetune` | warm-start fine-tuning on the most recent target data |
| `mas` | output-sensitivity regularisation toward the frozen parameters |
| `replay` | fine-tuning with bounded rehearsal of retained site data |

Hyper-parameters are those already used in the offline study (epochs 10, batch
64, lr 1e-4) and are **not** tuned during the collection window. The candidate
set is closed: no policy may be added after the first target day.

## 3. Endpoints

**Primary — realised controller cost.** For each candidate and each day: solve
the two-site MPC on that candidate's forecast, then settle the resulting plan
against measured demand and PV (`cost_optimised_realised_eur`). The plan is
never re-solved on the actuals.

**Secondary.**
1. Forecast nRMSE on the same days.
2. Gauge-corrected nRMSE, with the scalar fitted on a window strictly earlier
   than the evaluated day.
3. Planned-minus-realised savings, per day (`overstatement_eur`).

The controller formulation is the one fixed in `CONTROLLER_REFERENCE.md`
(pending). Until that file exists and is committed, collection may start but the
primary endpoint cannot be computed — the three repositories currently disagree
on `w_import`, `u_bounds` and `ramp_max`.

## 4. Pre-registered predictions

Stated now so they can fail.

- **P1.** The ranking of candidates by nRMSE will differ from their ranking by
  realised controller cost, on at least one adjacent pair.
  *Basis:* e15 (n=5) — replay best on retention nRMSE (0.418) and worst on the
  controller objective (1197.0). **Falsified if** the two rankings agree.
- **P2.** Planned savings will exceed realised savings on a majority of settled
  days. *Basis:* e18 — +1.69 EUR/day on real prices, one-directional.
  **Falsified if** the median daily overstatement is ≤ 0.
- **P3.** Gauge correction will change the ranking of `finetune` versus `mas`.
  *Basis:* e13 — raw p = 0.43, gauged p = 0.002 (n = 10 seeds, offline).
  **Falsified if** the ordering is the same before and after correction.

P1 is the prediction the paper rests on. P2 and P3 are supporting.

## 5. Analysis plan

Paired by day across candidates. Wilcoxon signed-rank on daily differences,
Holm correction across the candidate comparisons within each endpoint family,
two-sided, α = 0.05. Effect sizes reported as rank-biserial correlation with
bootstrap confidence intervals. Days with `solver_status` outside
{optimal, optimal_inaccurate}, or with incomplete telemetry, are excluded — and
the exclusion count is reported.

n is whatever the 90 days yield after exclusions. **No stopping rule keyed to
the result:** the window is not extended because a p-value is close, nor cut
short because it is satisfying.

## 6. No-peeking

Before the reading date, only these may be inspected: row counts, ingestion
gaps, solver status, and obvious data faults. Anything touching the primary
endpoint — per-candidate cost, ranking, any test — is off limits. If a bug
forces an early look, it is recorded here as an amendment with its date and
what was seen.

## 7. Known threats

- **Single site, single architecture.** See §1.
- **Prices.** Settlement uses day-ahead prices for the bidding zone; if the
  price series is unavailable for a day, that day is excluded rather than
  imputed.
- **Drift may simply not occur** in 90 days. Then the endpoints will be
  uninformative rather than negative, and that is the honest report.
- **Selection through the controller.** All candidates feed the *same*
  controller formulation; comparisons are valid only within that formulation
  and must not be stated as a general property of the methods.

## Amendments

_None._
