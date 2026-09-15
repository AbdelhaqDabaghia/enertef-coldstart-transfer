# Increasing telemetry freshness — proposal for discussion with Liviu

**Status: proposal only. No EventBridge schedule has been changed.**

Prepared 2026-09-15. Everything below is measured from the deployed Lambda
(`enertef-leneda-ingest`), its CloudWatch logs, and the EventBridge rules —
not estimated.

---

## 1. The problem, quantified

`enertef-svc1-15min` runs the forecast and MPC every 15 minutes — about 96
cycles a day. `enertef-leneda-ingest-15min` is scheduled `rate(1 day)`
(the rule name is wrong and should be corrected regardless of anything else
here). So ~96 forecast cycles per day run against telemetry that advances once.

Measured on 2026-09-15:

| | |
|---|---|
| ingest fires | daily, 08:55 UTC |
| rows written | 96 per asset per run (one day at 15-min resolution) |
| newest telemetry point at 08:42 UTC | 2026-09-13 23:45 |
| **staleness at that moment** | **32.9 h** |
| staleness range over a day | ~9 h (just after ingest) → ~33 h (just before) |

The forecaster's lookback is 672 steps anchored to *now*, so at 33 h staleness
roughly **132 slots — about 20 % of the window, and specifically the most
recent 20 % — are filled with a median constant**, including `lag_1`. See
`deploy/padding_counter.patch`.

---

## 2. Why simply polling more often does not work

The Lambda's own comment records the incident:

```python
# Leneda blocked our IP for requesting same-day data (2026-09-10, Liviu) --
# meter readings for "today" are not yet published on their side. Never ask
# past the start of the current UTC day; end the window at yesterday's
# boundary instead of "now".
```

**The block was triggered by the requested window, not by the request rate.**
The current code therefore ends every window at `00:00 today` and looks back
`WINDOW_HOURS = 26`.

That constraint, not the cadence, is what sets the staleness floor. If a
window may never extend past the start of the current UTC day, then the
freshest obtainable reading is always *yesterday 23:45*, and polling hourly
would return **the same rows 24 times a day** — no freshness gain, 24× the
request volume, and a reintroduced risk of the exact behaviour that caused the
block.

**So the thing to negotiate is the publication lag, not the polling rate.**

---

## 3. What we need from Liviu

One question decides everything else:

> **How long after a 15-minute interval ends is its meter reading reliably
> available through the EnergyPark proxy?**

Call that lag **L**. If readings settle within, say, 4 hours, a window ending
at `now − L` is safe and cuts staleness from 9–33 h to roughly L to L + the
polling interval.

Supporting questions, in order of importance:

1. **What exactly triggered the 2026-09-10 block?** Requesting timestamps
   beyond the published horizon, request volume, or both? The code assumes the
   first; we have never had that confirmed.
2. **Is there a documented or contractual rate limit** on the proxy —
   requests per minute, per hour, per day, per POD? We currently issue
   **3 requests/day** and have no idea what the ceiling is.
3. **Does the proxy return an error or an empty result** for a not-yet-published
   interval? If it returns empty rather than erroring, we can probe safely and
   measure L ourselves without risk.
4. **Is L stable**, or does it vary by meter, by day of week, or at month
   boundaries when readings may be revised?
5. **Are readings ever revised after first publication?** If so, we need to
   keep re-fetching a trailing window, not only the newest slice — this changes
   the design below.

---

## 4. Proposed design, once L is known

Two changes, both small:

**(a) End the window at `now − L − margin` instead of `00:00 today`.**
A `SAFETY_MARGIN_HOURS` on top of L, agreed with Liviu, absorbs jitter. This is
a one-line change to `lambda_handler` and is the change that actually delivers
freshness.

**(b) Poll at an interval matched to L, with a shorter window.**
There is no value in polling faster than the data publishes. A window of
`L + polling_interval + margin` is enough to catch everything new plus an
overlap for safety; the upsert is idempotent (`ON CONFLICT` on the existing
key), so overlap is harmless.

---

## 5. Volume, under each option

Each request covers one POD. Three PODs per run. A 15-minute resolution means
4 points per hour per POD.

| option | cadence | window | requests/day | points fetched/day | vs today |
|---|---|---|---|---|---|
| **current** | 1/day | 26 h | **3** | 312 | 1× |
| conservative | 4/day (6 h) | 8 h | 12 | 288 | **0.9×** |
| moderate | 8/day (3 h) | 5 h | 24 | 360 | 1.2× |
| aggressive | 24/day (1 h) | 3 h | 72 | 648 | 2.1× |
| naive "poll often, keep 26 h window" | 24/day | 26 h | 72 | 7 488 | **24×** |

The point of this table is the last row against the second. **Polling four
times as often with a correctly sized window moves *less* data than today**,
because the current 26-hour window re-fetches the same day repeatedly. The
naive version is 24× the volume for no freshness benefit at all.

A defensible opening position with Liviu is therefore the **conservative** row:
4 runs/day, 8-hour window, **12 requests/day against today's 3**, and slightly
*less* data transferred. That is a modest ask, and it reduces staleness from
9–33 h to roughly L + 6 h.

---

## 6. What we are *not* asking for

Worth saying explicitly, since the previous incident makes this a trust
conversation as much as a technical one:

- **No same-day data.** The window still ends before the published horizon,
  with an agreed margin. That rule is not being relaxed.
- **No real POD identifiers.** We continue to send only the placeholder ids
  (`…888881/2/3`) that the proxy translates internally. The code carries an
  explicit warning against ever substituting the real ones.
- **No increase in scope.** Same three metering points, same OBIS codes —
  `1-65:1.29.3` for the two consumption meters, `1-1:2.29.0` for PV
  (the production code Liviu confirmed on 2026-09, and the only candidate
  returning `calculated=false`).

---

## 7. Sequencing

1. Put questions §3.1–3.5 to Liviu; the answer to §3.3 may let us measure L
   ourselves at zero risk.
2. Agree L, the margin, and an acceptable cadence **in writing**, so the next
   person to touch this rule can see the constraint rather than rediscover it
   through another block.
3. Apply `deploy/padding_counter.patch` **first**, and emit `pad_fraction` as a
   CloudWatch metric. That gives a before/after measurement of exactly what the
   cadence change buys, instead of an assumption.
4. Change the window (§4a), then the cadence (§4b), as **two separate
   deployments**. If something goes wrong, one change at a time is diagnosable
   and two are not.
5. Rename `enertef-leneda-ingest-15min`. A rule whose name asserts a cadence it
   does not have is how this went unnoticed for months.

---

## 8. Honest caveat

Freshness fixes the padding; it does not fix the other two defects on this
feature path. The train/serve rolling-mean mismatch is addressed by the causal
retrain (`ev_cnn_lstm_causal_full_warm_e20_s1.keras`), and the
forecast-vs-realised KPI reporting by `scripts/settle_kpis.py`. All three are
independent, and fixing only this one will not by itself make the KPI pass.
