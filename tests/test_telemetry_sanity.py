"""
test_telemetry_sanity.py -- the physical checks, and the timezone trap.

The PV-at-night check is the only thing in the stack that can catch a POD/label
swap, because historical.telemetry_15min stores a label rather than a POD or a
foreign key. It is worth testing carefully, and in particular it is worth
pinning the mistake that made it fire falsely the first time: applying local
night hours to UTC timestamps, which at 49.5 N flags real generation at dawn
for three months of the year.

    python tests/test_telemetry_sanity.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from telemetry_sanity import (run_checks, check_pv_dark_at_night,
                              check_demand_non_negative, check_not_frozen,
                              check_reporting_completeness)      # noqa: E402


def _day(date="2026-06-21"):
    """A summer day in UTC -- the season where the timezone bug bites."""
    return pd.date_range("%s 00:00" % date, periods=96, freq="15min", tz="UTC")


def _solar(ts):
    """Generation that starts at dawn LOCAL and is zero in local night."""
    loc = ts.tz_convert("Europe/Luxembourg")
    h = loc.hour + loc.minute / 60.0
    v = np.clip(np.sin((h - 5.5) / 15.0 * np.pi), 0, None) * 45.0
    return np.where((h < 5.5) | (h > 20.5), 0.0, v)


def test_real_dawn_generation_is_not_flagged():
    """The false positive that the first version produced: PV at 05:45 local
    in June is legitimate, and lands at 03:45 UTC."""
    ts = _day()
    assert check_pv_dark_at_night(ts, _solar(ts)) is None, (
        "flagged genuine summer dawn generation -- the local/UTC confusion")


def test_pv_on_a_charging_meter_is_caught():
    """The failure mode the schema cannot detect: PV labelled onto a meter
    that draws power at night."""
    ts = _day()
    charging = np.full(96, 40.0)          # busy all night, as a charger can be
    msg = check_pv_dark_at_night(ts, charging)
    assert msg is not None and "NOT DARK" in msg


def test_inverter_standby_is_tolerated():
    ts = _day()
    pv = _solar(ts) + 0.3                 # below the 1 kW tolerance
    assert check_pv_dark_at_night(ts, pv) is None


def test_negative_demand_is_caught():
    msg = check_demand_non_negative("EMOB1", np.full(96, -12.0))
    assert msg is not None and "NEGATIVE" in msg


def test_frozen_meter_is_caught():
    """A stuck meter is indistinguishable from an idle site in every aggregate."""
    msg = check_not_frozen("EMOB2", np.full(96, 17.0))
    assert msg is not None and "FROZEN" in msg


def test_a_genuinely_idle_but_live_meter_is_not_frozen():
    """Idle is not the same as stuck: small variation must pass."""
    r = np.random.default_rng(0)
    assert check_not_frozen("EMOB2", r.normal(0.5, 0.2, 96)) is None


def test_partial_day_is_caught():
    v = np.full(96, 20.0)
    v[40:] = np.nan
    msg = check_reporting_completeness("EMOB1", v)
    assert msg is not None and "ONLY" in msg


def test_a_healthy_day_passes_every_check():
    ts = _day()
    r = np.random.default_rng(1)
    problems = run_checks(ts, {
        "PV": _solar(ts),
        "EMOB1": np.abs(r.normal(40, 15, 96)),
        "EMOB2": np.abs(r.normal(60, 20, 96)),
    })
    assert not problems, problems


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
