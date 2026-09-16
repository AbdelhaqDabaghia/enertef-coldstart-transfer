# Where this project stands

**Read this first.** It is the fastest way to resume work on another machine or
in a new session. The commit log carries the reasoning; this file carries the
conclusions.

Last updated: 2026-09-15.

---

## What this repository is now

It started as the reproducible pipeline for a continual-learning transfer paper.
It is now two things:

1. **The paper** (`paper/v4/main.tex`) — a deployment study of continual
   learning at a Luxembourg commercial energy site.
2. **A deployment audit** (`deploy/`, `scripts/e19`–`e21`, `tests/`) — the
   defects that audit found in the live Service-1 system, and the fixes.

The second grew out of the first. Checking whether the paper's claims survived a
correct feature pipeline exposed a production model that was serving roughly a
quarter of real demand.

---

## Conclusions that are established

Each is traceable to a CSV under `Data/results/`.

### The paper

- **The gauge-decomposition result does not survive causal features.** Under the
  leaky pipeline, correcting output amplitude made the regulariser beat plain
  fine-tuning (p = 0.002). Causally there is no difference (p = 0.43). This was
  Contribution 1 and it is withdrawn. `e13`, n = 10.
- **Rehearsal dominates causally.** Bounded replay beats fine-tuning by 0.346
  nRMSE, unanimous over 10 seeds, d_z = −17.57, and is the only mechanism that
  beats naive persistence. The regulariser is indistinguishable from plain
  fine-tuning. This restores the direction of the conference paper, far more
  decisively. `e13`.
- **Forecast accuracy does not predict decision value.** Rehearsal forecasts
  34 % better on 35/35 day-seed pairs, yet realised controller cost is
  statistically unchanged (p = 0.645). This is a *decoupling*, not an
  inversion — state the weaker claim. `e16`.
- **No headroom for online arbitration.** An oracle selecting daily with
  knowledge of the outcome beats the best fixed choice by 0.41 %. `e16`, `m1`.

### The deployment

- **Train/serve feature mismatch.** The model was fitted on rolling means
  containing y_t and is served means computed a third way. Same weights, same
  holdout: nRMSE 0.548 → 1.389, mean prediction 41.8 → 14.5 kW against a true
  56.9. `e19`.
- **A causal refit fixes most of it.** nRMSE 0.8495, bias −4.2 kW, mean
  prediction 52.7 kW. Error down 39 %, bias down 90 %. `retrain_causal`.
- **The KPI was never settled against measurement.** Planned 11.5 % vs
  simulated-on-real-data 6.0 % over 21 days. `e18`. See the warning below about
  the word "realised".
- **The controller never actuated anything until 2026-09-16 06:19 UTC.** The
  MQTT publish to the chargers failed on every cycle — 400 consecutive failures
  over the preceding fortnight, zero successes — so plans were computed, KPIs
  claiming savings were written, and nothing reached the chargers. EMOB2 was
  additionally never commanded at all, even in the code path. Fixed upstream in
  svc1-runner (93bebab, 3028175). **This is the single most consequential
  finding in the project.**
- **The deployed controller was running on a flat price.** The ENTSO-E token
  had been returning 401 since 2026-07-02, so `contextual.prices` was stale and
  `fetch_prices` fell back to a constant 100 EUR/MWh. At a flat price an
  energy-conserving shift *cannot* change the reported bill, so savings were
  exactly 0.00 EUR every cycle. **Fixed 2026-09-15.** `e21`.
- **Telemetry is stale and the gap is fabricated.** Ingest runs once a day; the
  lookback is anchored to `now`; every missing slot is filled with the median.
  At 33 h staleness that was ~20 % of the window, including `lag_1`. Partly
  mitigated 2026-09-15 by re-pinning the ingest to 00:05 UTC.

---

## Conclusions that were reached and later found wrong

Recorded so nobody re-derives them. Each was stated confidently before being
measured.

| claim | reality |
|---|---|
| "The ENTSO-E API returns 404" | It returns **401**. The 404/timeout seen locally is the LIST firewall blocking `web-api.tp.entsoe.eu`: TCP connects, then zero bytes, while `github.com` returns 200 from the same shell. Local tests of that host are not diagnostic. |
| "The MPC solver is failing" | SLSQP converges 7/7 days and matches cvxpy/OSQP to 0.002 EUR/day. The solver and formulation are sound. `e21`. |
| "The production feature builder is causal" | It is not. `compute_lag_features` takes `[idx-w+1, idx+1)`, inclusive. The `if end > n` line is a bounds guard, not a causality shift. |
| "serve = (w-1)/w × causal" | Wrong; the windows differ. The true relation is `serve = causal − y[t-w]/w`. |
| "Replay's retention advantage is entirely gauge" | About 30 % was amplitude; 70 % is structural. |
| "`mpc_runs` is the KPI table" | Nothing writes it. The live path is `historical.kpi_validation`. |
| "1.74 EUR/day is the value of forecasting" | Wrong arm of the experiment, and superseded by `e18`. |

**"Realised" is the wrong word, and it is used throughout.** Every euro figure
this project has reported as a *realised* saving — `e18`'s 7.86 EUR/day
included — applies the planned deviation `u` to the demand that actually
arrived. But `u` was never dispatched before 2026-09-16 06:19 UTC, so that
demand already reflects no control. Those figures are **counterfactuals**: what
the site would have saved had the plan been applied. It was not applied. The
site saved nothing.

They are more honest than forecast-against-forecast, because the demand and
prices are real. But they are simulations on real data and must be described
that way. Read "realised" in `e18`, `e20`, `e21` and in `paper/v4` as
**"simulated under realised conditions"**. Settlement only becomes meaningful
from the 06:19 boundary onwards.

**A second standing caveat:** `e19`/`e20` evaluated against a contiguous CSV, so
they exclude the stale-history padding defect. `e22` measured what that costs:
+0.096 nRMSE for the causal model at the expected mean staleness of 12.4 h.
Real, worth carrying, not large enough to overturn anything concluded.

---

## What is deployed vs. what is written but not applied

**Applied to production (2026-09-15):**
- Valid ENTSO-E token in Secrets Manager `enertef/entsoe-token`.
- CloudWatch alarms `enertef-price-fetcher-errors` and
  `-not-running`, on SNS topic `enertef-alerts` (**no subscriber yet**).
- `enertef-leneda-ingest-15min` re-pinned `rate(1 day)` → `cron(5 0 * * ? *)`.
  Same cadence, same window, same request count.

**Applied upstream in `svc1-runner`** (by another session, on top of the
baseline commit; `f113630`, `d395ac0`, `4ea5333` committed but not yet shipped
as of 2026-09-16):
- `e69817e` — baseline: the feature builders, which had never been committed.
- `0a17f7f` — padding counter (EV half).
- `5adaa58` — `EV_FIX_B1A` gates the causal serving convention on the same flag
  as the model pointer, so they flip together in one coordinated deploy.
- `93bebab`, `3028175` — IoT publish fixed; both chargers now actuate.
- `f113630` — `mpc_setpoints` written per charger.
- `4ea5333` — the KPI row split (`cost_reduction_pct_planned` judged on 20 %,
  `savings_eur_planned` on 5 EUR). This supersedes what was
  `deploy/kpi_reporting.patch`, now deleted.

**Written, tested, NOT applied** (all in `deploy/`):
- `price_fallback.patch` — records IDLE rather than FAIL when prices are flat.
- `padding_counter.patch` — the PV half; the EV half is applied upstream.
- `validation_v2.py` — promotion gate with an absolute persistence floor.
- `feature_mode_guard.py` — refuses to serve a model against features it was
  not fitted on.
- `rehearsal_config.md` — switching to rehearsal is three env vars, no code.
- `ingestion_cadence_proposal.md` — for the Liviu conversation.

**The "three divergent copies" concern is resolved.** There were only two:
`enertef-svc1-clean` and `enertef-svc1-source/task` are byte-identical, and both
are stale (2 Jul). The deployed lineage is `svc1-runner`, whose file timestamp
matches the ECR image push to within two minutes. The two versions differ by 19
lines, functionally only a `MIN_BASELINE_EUR_FOR_PCT` divide-by-zero guard. Do
not patch the stale copies.

**Deployment candidate:** `Data/models/ev_cnn_lstm_causal_full_warm_e20_s1.keras`
with its paired `ev_scalers_causal_full_warm_e20_s1.joblib` — they must ship
together, the frozen production scalers will not work.

---

## Open items

1. **Run `scripts/settle_kpis.py` inside the VPC.** The single step that turns
   the planned KPI into a defensible realised one. Everything else is ready.
2. **Rotate the ENTSO-E token** — it passed through a chat transcript.
3. **Subscribe an address** to `enertef-alerts`, or the alarms fire into nothing.
4. **Decide which `realtime_runner.py` copy is deployed**, then apply the patches.
5. **Item 6 of the plan** — a promotion gate on realised control cost. Blocked
   deliberately: the threshold must come from the settled distribution, not be
   invented.
6. **Item 8** — prospective evaluation. `e20` is a *holdout*, not prospective:
   the window is the last 7 days of the training series. Genuine prospective
   evaluation needs data collected after the models were frozen, pulled from
   the live database.
7. **Paper**: `paper/v4/main.tex` has never been compiled (no LaTeX toolchain
   here). Two citations, `li2024building` and `ng2024cost`, are unverified.
   Section VIII-B states a conclusion of the conference version does not hold —
   that needs a co-author conversation before submission.

---

## Resuming on another machine

```bash
git clone https://github.com/AbdelhaqDabaghia/enertef-coldstart-transfer.git
cd enertef-coldstart-transfer
cat STATE.md                 # this file
git log --oneline | head -30 # the reasoning, newest first
python tests/run_all.py      # 7 files, 51 tests, no GPU or DB needed
```

`tests/run_all.py` is the fastest way to confirm the environment is sane: every
test corresponds to a defect that reached production, so a green run means the
guards are intact.

Commit messages are written to be read by someone who was not present. Where a
conclusion was corrected, the commit that corrected it says so explicitly —
`git log --grep="correct"` finds them.

**What is NOT in this repository:** AWS credentials, the ENTSO-E token, database
passwords, real POD identifiers, and the deployed `realtime_runner.py`. The
repository is public.
