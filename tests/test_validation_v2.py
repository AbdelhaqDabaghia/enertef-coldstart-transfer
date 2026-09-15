"""
test_validation_v2.py -- the gate must reject what the old one waved through.

Every test is built from a situation this project actually measured, not from
a hypothetical, so a failure here means a real deployment path has reopened.

    python tests/test_validation_v2.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "deploy")
from validation_v2 import (validate_new_model, persistence_forecast,
                           compute_metrics)                # noqa: E402

rng = np.random.default_rng(0)
# a plausible EV day: quiet nights, busy daytime
TRUE = np.abs(np.concatenate([rng.normal(10, 3, 32),
                              rng.normal(90, 25, 48),
                              rng.normal(25, 8, 16)]))


def _noisy(scale=1.0, noise=8.0, seed=1):
    r = np.random.default_rng(seed)
    return np.clip(TRUE * scale + r.normal(0, noise, len(TRUE)), 0, None)


def test_the_deployed_situation_is_rejected():
    """The real one. A candidate close to a poor incumbent, both beaten by a
    one-step lag. The old gate promoted this; the new one must not."""
    incumbent = _noisy(scale=0.26, seed=2)      # the 14.5/56.9 ratio
    candidate = _noisy(scale=0.27, seed=3)      # marginally 'better' MAE
    r = validate_new_model(incumbent, candidate, TRUE)
    assert not r["promote"], "promoted a model beaten by persistence"
    assert ("ABSOLUTE FLOOR" in r["reason"] or "SCALE" in r["reason"]), r["reason"]


def test_a_good_model_is_still_promoted():
    """The gate must not simply refuse everything."""
    incumbent = _noisy(scale=1.0, noise=22.0, seed=4)
    candidate = _noisy(scale=1.0, noise=6.0, seed=5)
    r = validate_new_model(incumbent, candidate, TRUE)
    assert r["promote"], r["reason"]
    assert "PROMOTE" in r["reason"]


def test_persistence_floor_binds_even_against_a_terrible_incumbent():
    """The core gap: an incumbent so bad that anything looks like progress."""
    awful = np.zeros_like(TRUE)
    pers_mae = compute_metrics(TRUE, persistence_forecast(TRUE))["mae"]
    # deliberately just the wrong side of the floor
    candidate = _noisy(scale=1.0, noise=pers_mae * 2.2, seed=6)
    r = validate_new_model(awful, candidate, TRUE)
    if compute_metrics(TRUE, candidate)["mae"] >= pers_mae:
        assert not r["promote"], "a sub-persistence candidate was promoted"


def test_degenerate_baseline_no_longer_auto_promotes_garbage():
    """validation.py returned promote=True unconditionally here."""
    candidate = _noisy(scale=0.25, seed=7)      # useless
    r = validate_new_model(np.full(len(TRUE), np.nan), candidate, TRUE)
    assert not r["promote"], (
        "a degenerate baseline still promotes anything: %s" % r["reason"])


def test_degenerate_baseline_promotes_a_genuinely_good_model():
    """But it must not block a good candidate just because the baseline broke."""
    candidate = _noisy(scale=1.0, noise=5.0, seed=8)
    r = validate_new_model(np.full(len(TRUE), np.nan), candidate, TRUE)
    assert r["promote"], r["reason"]
    assert "degenerate" in r["reason"]


def test_scale_check_catches_a_half_scale_model():
    """A model with acceptable MAE but a systematic amplitude error."""
    incumbent = _noisy(scale=1.0, noise=30.0, seed=9)
    candidate = _noisy(scale=0.4, noise=2.0, seed=10)
    r = validate_new_model(incumbent, candidate, TRUE)
    assert not r["promote"]
    assert "SCALE" in r["reason"] or "FLOOR" in r["reason"], r["reason"]


def test_empty_holdout_refuses_rather_than_assumes():
    r = validate_new_model(np.array([]), np.array([]), np.array([]))
    assert not r["promote"] and "Empty holdout" in r["reason"]


def test_invalid_predictions_rejected():
    r = validate_new_model(_noisy(seed=11), np.full(len(TRUE), np.nan), TRUE)
    assert not r["promote"] and "invalid" in r["reason"].lower()


def test_checks_can_be_disabled_for_backward_compatibility():
    """Callers that want the old behaviour must be able to ask for it."""
    incumbent = _noisy(scale=0.26, seed=12)
    candidate = _noisy(scale=0.27, seed=13)
    r = validate_new_model(incumbent, candidate, TRUE,
                           require_beat_persistence=False, scale_bounds=None)
    assert r["promote"], "disabling the new checks did not restore the old path"


def test_result_reports_what_it_judged_against():
    """An operator reading the log should see the floor, not just the verdict."""
    r = validate_new_model(_noisy(seed=14), _noisy(noise=5.0, seed=15), TRUE)
    assert "persistence_metrics" in r and "scale_ratio" in r


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
