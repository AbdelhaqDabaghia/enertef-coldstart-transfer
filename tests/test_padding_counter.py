"""
test_padding_counter.py -- the helper from deploy/padding_counter.patch.

The patch cannot be applied from here (the builders live in another repo, in
several copies), so the logic is pinned here instead. If someone applies the
patch, these tests should move with it.

The case that matters is the measured one: telemetry 33 h stale against a
672-slot lookback anchored to now, which pads the most recent ~132 slots with
a median constant and is invisible to the existing row-count guard.

    python tests/test_padding_counter.py
"""
from __future__ import annotations
import sys
import logging
from datetime import datetime, timedelta, timezone
import numpy as np

logger = logging.getLogger("padding_test")
logging.basicConfig(level=logging.CRITICAL)      # keep test output quiet


def fill_series_from_history(series_dict, start_time, n_slots, pad_value,
                             label, step_minutes=15, max_pad_fraction=None):
    """Verbatim from deploy/padding_counter.patch."""
    values = np.zeros(n_slots, dtype=np.float32)
    missing = np.zeros(n_slots, dtype=bool)
    for i in range(n_slots):
        ts = start_time + timedelta(minutes=step_minutes * i)
        if ts in series_dict:
            values[i] = series_dict[ts]
        else:
            values[i] = pad_value
            missing[i] = True

    n_padded = int(missing.sum())
    tail = 0
    for m in missing[::-1]:
        if not m:
            break
        tail += 1

    newest_real = max(series_dict) if series_dict else None
    window_end = start_time + timedelta(minutes=step_minutes * (n_slots - 1))
    staleness_h = (None if newest_real is None
                   else (window_end - newest_real).total_seconds() / 3600.0)

    stats = {"n_padded": n_padded, "n_slots": n_slots,
             "pad_fraction": n_padded / float(n_slots),
             "tail_padded": tail, "newest_real": newest_real,
             "staleness_hours": staleness_h}

    if max_pad_fraction is not None and stats["pad_fraction"] > max_pad_fraction:
        raise ValueError("%s: %.1f %% padded, above limit %.1f %%"
                         % (label, 100 * stats["pad_fraction"],
                            100 * max_pad_fraction))
    return values, stats


NOW = datetime(2026, 9, 15, 8, 45, tzinfo=timezone.utc)
N = 672
START = NOW - timedelta(minutes=15 * N)


def _history(up_to, n=N):
    """Contiguous 15-min history ending at `up_to`."""
    return {up_to - timedelta(minutes=15 * i): 40.0 + i % 7 for i in range(n)}


def test_fresh_history_pads_nothing():
    d = _history(START + timedelta(minutes=15 * (N - 1)))
    _, s = fill_series_from_history(d, START, N, 99.0, "fresh")
    assert s["n_padded"] == 0, s
    assert s["tail_padded"] == 0
    assert abs(s["staleness_hours"]) < 1e-9


def test_the_measured_case_33h_stale():
    """The real one. Newest point 2026-09-13 23:45 against a window ending
    2026-09-15 08:30 -- what production actually had."""
    newest = datetime(2026, 9, 13, 23, 45, tzinfo=timezone.utc)
    d = _history(newest)
    _, s = fill_series_from_history(d, START, N, 42.0, "stale")
    # 32.75 h of 15-min slots = 131 slots
    assert 125 <= s["tail_padded"] <= 140, s["tail_padded"]
    assert 0.18 <= s["pad_fraction"] <= 0.22, s["pad_fraction"]
    assert 32 <= s["staleness_hours"] <= 34, s["staleness_hours"]


def test_tail_padding_is_reported_separately():
    """Padding at the end replaces lag_1; padding in the past does not. The
    two must be distinguishable or the log cannot convey severity."""
    d = _history(START + timedelta(minutes=15 * (N - 1)))
    for i in range(10, 30):                       # a hole in the MIDDLE
        d.pop(START + timedelta(minutes=15 * i), None)
    _, s = fill_series_from_history(d, START, N, 42.0, "hole")
    assert s["n_padded"] == 20, s["n_padded"]
    assert s["tail_padded"] == 0, (
        "a gap in the distant past was counted as tail padding")


def test_lag_1_is_the_median_when_the_tail_is_padded():
    """The concrete harm, asserted rather than described."""
    newest = NOW - timedelta(hours=33)
    d = _history(newest)
    vals, s = fill_series_from_history(d, START, N, 42.0, "stale")
    assert vals[-1] == 42.0, "the most recent slot should be the pad value"
    assert vals[-2] == 42.0


def test_empty_history_does_not_crash():
    vals, s = fill_series_from_history({}, START, N, 0.0, "empty")
    assert s["n_padded"] == N and s["tail_padded"] == N
    assert s["newest_real"] is None and s["staleness_hours"] is None


def test_refusal_is_opt_in():
    """No threshold by default -- none has been measured."""
    newest = NOW - timedelta(hours=33)
    d = _history(newest)
    fill_series_from_history(d, START, N, 42.0, "stale")       # must not raise
    try:
        fill_series_from_history(d, START, N, 42.0, "stale",
                                 max_pad_fraction=0.10)
    except ValueError:
        return
    raise AssertionError("an explicit limit of 10 % was not enforced")


def test_row_count_guard_cannot_see_this():
    """Why the existing `if len(ev_history) < 96` check is blind: the query
    returns a FULL 672 rows and 20 % of slots are still padded."""
    newest = NOW - timedelta(hours=33)
    d = _history(newest, n=N)
    assert len(d) == N, "the query returned a full window"
    _, s = fill_series_from_history(d, START, N, 42.0, "stale")
    assert s["n_padded"] > 100, (
        "a full row count coexisting with heavy padding is the whole point")


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
