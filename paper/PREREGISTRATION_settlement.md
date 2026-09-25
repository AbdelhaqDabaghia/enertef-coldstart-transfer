# Pre-registration — settled savings of the deployed controller

**Status: written before any settleable day exists under this protocol.** The
first day it admits is 2026-09-26. Nothing below may be changed once the first
day is written; changes go in a dated amendment section and every amendment is
reported in the paper.

- Registered on: _the commit date of this file IS the registration date_
- Protocol opens: **2026-09-26 00:00 UTC**
- Minimum window: **14 admissible days** (see §3)
- Reading date: the day after the 14th admissible day. No analysis of the
  primary endpoint before then.
- The window **remains open** after the minimum is reached; whatever is
  complete at submission is reported as the pre-registered result, and the
  fuller window is reported in any revision.

This is a separate protocol from `PREREGISTRATION_prospective.md`, which fixes
a 90-day comparison of adaptation policies. That one is unaffected.

---

## 1. What is being measured, and why it needs registering

Three quantities are computed from the same dispatched plan:

- **planned** — optimised cost minus unmanaged cost, both on the forecast;
- **counterfactual** — that plan settled against metered demand on a day it was
  not dispatched;
- **settled** — that plan dispatched, and settled against metered demand and
  the day-ahead price that applied.

Retrospectively, over 21 days on real ENTSO-E prices, planned averaged
+9.55 EUR/day (11.5 % of the bill) and settled +7.86 EUR/day (6.0 %): the
relative figure overstated by nearly a factor of two. That measurement was made
after the data were visible, which is exactly the objection this protocol
removes.

## 2. Endpoints and decision rules

**Primary.** The per-day difference `planned − settled`, tested against zero
with a two-sided Wilcoxon signed-rank test at α = 0.05, paired by day. Effect
size reported as paired `d_z` with a 95 % bootstrap interval. The pairing is
exact: same day, same plan, same prices — only the reference the plan is
settled against differs.

**Secondary.** Settled savings tested against zero, same test, same α. No
correction between the two: they are reported as one primary and one secondary,
not as a family.

**Descriptive, no test.** Per-day settled savings, the unmanaged bill, the
intraday price spread, the fraction of the lookback imputed, and the number of
cycles in which setpoints were accepted by each charger.

Power, computed from the 21-day retrospective sample (SD of the paired
difference 1.27 EUR/day): n = 5 suffices to detect the observed 1.69 EUR/day
gap at 80 % power. Fourteen days is therefore not chosen for power. It is
chosen so the window spans **two complete weekly cycles**, because the same
sample shows week-to-week variation of 23 % in settled savings
(9.12 / 7.46 / 7.01 EUR/day over three weeks) and a single week cannot
distinguish the site's behaviour from that week's.

## 3. Which days count — fixed in advance

A day is **admissible** only if all four hold for every cycle of that day. Each
corresponds to a failure already observed on this system, and each is checked
from the record, not by judgement:

1. **The setpoints were dispatched and the database recorded it.** Rows carry
   `status = 'sent'` with per-charger granularity. Days before
   2026-09-16 06:33:52 UTC cannot satisfy this and are outside the protocol.
2. **The price applied was a market price.** The horizon price is not strictly
   constant. A constant tariff makes an energy-conserving shift worthless by
   construction, so a settled figure over such a day is exactly zero regardless
   of the controller — it is a non-measurement, not a zero result.
3. **Metered demand is present** for the settled period, with any imputed value
   marked as imputed rather than returned as data.
4. **The PV forecast was produced by the model**, not by the fallback proxy
   that engages when weather is unavailable.

A day failing any condition is **excluded as a non-measurement and the window
is extended by one day**. Exclusions are counted and reported with their
reason; the paper states how many days were excluded and why.

This rule is written now precisely because 2026-09-25 would fail conditions 2
and 4 — a 30-second ENTSO-E read timeout the previous afternoon left no prices,
and an Open-Meteo 503 left no weather. Deciding after the fact which days to
drop is the thing pre-registration exists to prevent.

## 4. Interim look

One interim analysis after **7 admissible days**, on the **variance only**. If
the observed SD of the paired difference exceeds the retrospective estimate by
more than 50 %, the minimum window extends to 21 days; the extension is
recorded with its date. The interim does **not** examine the effect, and the
window does not stop early for a favourable result: looking at the effect
mid-stream inflates the false-positive rate and would invalidate the primary
endpoint.

## 5. What would falsify the expected result

The expectation is that planned exceeds settled. Three outcomes are possible
and all three will be reported:

- **planned > settled**, confirming the retrospective measurement;
- **no detectable difference**, which would mean the retrospective gap was an
  artefact of the days it was measured on;
- **settled > planned**, which would mean the forecast systematically
  under-predicts in a way that favours the controller, and would require
  explaining rather than reporting.

## 6. Configuration frozen for the window

The deployed configuration is held fixed for the duration: EV and PV
forecasters adapted nightly at 06:30 UTC under the output-sensitivity
regulariser (λ = 1000, no replay buffer), promotion gate at tolerance 1.05, the
two-site convex MPC with its deployed objective, 15-minute cycles.

Any change to this configuration during the window ends it. If a change is
operationally necessary, the window closes at the last admissible day before
the change and a new window opens after it, both reported.

## 7. Commitment

The result is reported whatever its sign or significance, in the paper this
protocol was written for, with the per-day data released alongside.
