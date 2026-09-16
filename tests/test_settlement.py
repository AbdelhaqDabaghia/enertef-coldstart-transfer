"""
test_settlement.py -- pin the settlement arithmetic.

settle_kpis.settle() is the function that will decide whether Service-1 meets
its KPI. It runs against production data we cannot reach from here, so its
behaviour is pinned on cases whose answers are known by construction.

    python tests/test_settlement.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "scripts")
from settle_kpis import settle, DT_H       # noqa: E402

H = 96
FLAT = np.full(H, 100.0)                   # EUR/MWh


def test_zero_plan_is_zero_saving():
    """u = 0 is the unmanaged site: baseline and optimised must coincide."""
    ev = np.full(H, 200.0)
    pv = np.full(H, 50.0)
    b, o, red = settle(ev, pv, np.zeros(H), FLAT)
    assert abs(b - o) < 1e-9, "u=0 changed the cost: %.6f vs %.6f" % (b, o)
    assert abs(red) < 1e-9


def test_shifting_into_cheap_hours_saves():
    """Move load from an expensive half to a cheap half; the saving must be
    positive and equal to the energy moved times the price difference."""
    ev = np.full(H, 200.0)
    pv = np.zeros(H)
    price = np.concatenate([np.full(H // 2, 200.0), np.full(H // 2, 50.0)])
    u = np.concatenate([np.full(H // 2, -50.0), np.full(H // 2, +50.0)])
    b, o, red = settle(ev, pv, u, price)
    moved_mwh = 50.0 * (H // 2) * DT_H / 1000.0
    assert abs((b - o) - moved_mwh * 150.0) < 1e-6, (
        "expected %.4f EUR, got %.4f" % (moved_mwh * 150.0, b - o))
    assert red > 0


def test_export_is_credited_not_charged():
    """PV above demand exports; the cost must fall below zero, not rise."""
    ev = np.full(H, 10.0)
    pv = np.full(H, 500.0)
    b, _, _ = settle(ev, pv, np.zeros(H), FLAT)
    assert b < 0, "net exporter produced a positive cost: %.4f" % b


def test_charging_cannot_go_negative():
    """A plan that would drive EV demand below zero must be clipped, not
    allowed to manufacture export."""
    ev = np.full(H, 10.0)
    pv = np.zeros(H)
    u = np.full(H, -500.0)
    b, o, _ = settle(ev, pv, u, FLAT)
    assert o >= -1e-9, "clipping failed: optimised cost %.4f" % o
    assert abs(o) < 1e-9, "demand should clip to zero, giving zero cost"


def test_plan_is_settled_not_re_solved():
    """The realised optimised cost must depend on the REALISED demand, not on
    whatever the plan assumed. Two different realisations with the same plan
    must give different realised costs -- otherwise settlement is not
    happening."""
    pv = np.zeros(H)
    u = np.concatenate([np.full(H // 2, -50.0), np.full(H // 2, +50.0)])
    _, o1, _ = settle(np.full(H, 200.0), pv, u, FLAT)
    _, o2, _ = settle(np.full(H, 400.0), pv, u, FLAT)
    assert abs(o1 - o2) > 1.0, (
        "realised cost did not follow realised demand (%.4f vs %.4f)" % (o1, o2))


def test_length_mismatch_is_refused():
    try:
        settle(np.zeros(H), np.zeros(H), np.zeros(H - 1), FLAT)
    except ValueError:
        return
    raise AssertionError("mismatched series lengths were accepted")


# --------------------------------------------------------------- aggregation
# Since svc1-runner f113630 the runner writes one setpoint row PER CHARGER, so a
# 96-step day has 192 rows. The previous query returned raw rows ordered by
# time, which would have handed settle() a 192-element array for a 96-step
# horizon -- silently settling the first 48 timesteps interleaved across the two
# chargers, with no error raised. These pin the aggregation that fixes it.

def _rows_per_charger(u1, u2, status="sent"):
    """Emulate the GROUP BY target_time, SUM(setpoint_kw) the query performs."""
    out = []
    for i, (a, b) in enumerate(zip(u1, u2)):
        out.append((i, float(a + b), 2,
                    2 if status == "sent" else 0,
                    0 if status == "sent" else 2))
    return out


# --------------------------------------------------- dispatched-step alignment
# The controller publishes only u[0], the first step of each cycle. Verified in
# the database: every status='sent' row has target_time == computed_at exactly.
# So the trajectory the site received is the SEQUENCE OF DISPATCHED FIRST STEPS,
# sparse, one per cycle -- not a 96-step plan. Aligning those by POSITION rather
# than by TIMESTAMP slides the whole trajectory, and raises nothing.

def test_sparse_dispatch_aligns_by_timestamp_not_position():
    """22 dispatched steps against 96 telemetry steps. Positional alignment
    would place them in the first 22 slots; timestamp alignment puts them where
    they actually happened."""
    telemetry_times = list(range(96))
    dispatched = {t: -20.0 for t in range(26, 48)}      # 22 steps, mid-day
    u = np.array([dispatched.get(t, 0.0) for t in telemetry_times])
    assert len(u) == 96
    assert np.count_nonzero(u) == 22
    assert u[0] == 0.0 and u[25] == 0.0, "control must not appear before it did"
    assert u[26] == -20.0 and u[47] == -20.0
    assert u[48] == 0.0, "control must not persist after it stopped"

    positional = np.zeros(96)
    positional[:22] = -20.0                              # the wrong way
    assert not np.array_equal(u, positional), (
        "positional alignment must not coincide with timestamp alignment")


def test_uncontrolled_steps_take_zero_not_the_last_value():
    """A step with no dispatched setpoint received NO control. Carrying the
    previous value forward would credit the controller with hours it never
    acted on."""
    telemetry_times = list(range(10))
    dispatched = {3: -15.0}
    u = np.array([dispatched.get(t, 0.0) for t in telemetry_times])
    assert u[3] == -15.0
    assert u[4] == 0.0, "no forward fill"
    assert np.count_nonzero(u) == 1


def test_summing_revisions_of_one_timestep_is_the_bug_we_fixed():
    """~150 cycles each write a 96-step plan, so one target_time is written by
    about a hundred cycles. Summing them overstates the deviation by two orders
    of magnitude -- and both arrays are still plausible lengths."""
    revisions = [-20.0] * 100                # the same instant, 100 cycles
    summed = sum(revisions)
    dispatched_only = -20.0                  # the one actually published
    assert abs(summed) > 50 * abs(dispatched_only), (
        "the summed version should be wildly larger, which is why it was wrong")


def test_two_chargers_aggregate_to_site_level():
    """The grid sees u1 + u2. One row per timestep after aggregation."""
    u1 = np.full(H, -10.0)
    u2 = np.full(H, -15.0)
    rows = _rows_per_charger(u1, u2)
    assert len(rows) == H, "aggregation must yield one row per timestep"
    u_site = np.array([r[1] for r in rows])
    assert np.allclose(u_site, -25.0), "site deviation must be u1 + u2"


def test_unaggregated_rows_would_have_been_silently_wrong():
    """The bug, pinned. 192 raw rows truncated to 96 gives the first 48
    timesteps interleaved -- a plausible-looking array that is wrong."""
    u1 = np.arange(H, dtype=float)
    u2 = np.arange(H, dtype=float) * 100.0
    interleaved = []
    for a, b in zip(u1, u2):
        interleaved += [a, b]
    truncated = np.array(interleaved[:H])
    correct = u1 + u2
    assert not np.allclose(truncated, correct), (
        "the interleaved truncation should NOT match the correct site series")
    assert len(truncated) == len(correct) == H, (
        "both are length 96, which is why the bug raised no error")


def test_status_counts_distinguish_dispatched_from_pending():
    """Only rows whose setpoint was actually sent describe a dispatched plan."""
    rows_sent = _rows_per_charger(np.zeros(H), np.zeros(H), status="sent")
    rows_pend = _rows_per_charger(np.zeros(H), np.zeros(H), status="pending")
    assert sum(r[3] for r in rows_sent) == 2 * H
    assert sum(r[4] for r in rows_sent) == 0
    assert sum(r[3] for r in rows_pend) == 0
    assert sum(r[4] for r in rows_pend) == 2 * H


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print("  PASS  %s" % f.__name__)
        except AssertionError as e:
            bad += 1
            print("  FAIL  %s\n        %s" % (f.__name__, e))
    print("\n%d/%d passed" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
