"""
test_feature_mode_guard.py -- the guard must fire on the real defect.

The point of the guard is that it would have caught what actually happened. So
the central test reconstructs that exact situation -- a model declaring
'causal', a builder serving 'serve' -- and asserts a refusal.

    python tests/test_feature_mode_guard.py
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, "deploy")
from feature_mode_guard import (assert_feature_mode, assert_scaler_pairing,
                                warn_if_prediction_scale_drifts,
                                FeatureModeMismatch)      # noqa: E402


def _meta(tmp, mode="causal", scalers="ev_scalers_causal.joblib"):
    p = os.path.join(tmp, "meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"feature_mode": mode, "scalers": scalers}, f)
    return p


def test_the_actual_defect_is_refused():
    """A causal model served the production 'serve' features -- the situation
    that under-predicted this site by 3.9x."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _meta(tmp, "causal")
        try:
            assert_feature_mode(p, builder_mode="serve")
        except FeatureModeMismatch as e:
            assert "MISMATCH" in str(e)
            return
        raise AssertionError("the guard accepted the defect it exists to catch")


def test_matching_convention_passes():
    with tempfile.TemporaryDirectory() as tmp:
        m = assert_feature_mode(_meta(tmp, "causal"), builder_mode="causal")
        assert m["feature_mode"] == "causal"


def test_missing_metadata_is_refused():
    try:
        assert_feature_mode("/nonexistent/meta.json", builder_mode="causal")
    except FeatureModeMismatch:
        return
    raise AssertionError("a model with no declared convention was accepted")


def test_scaler_pairing_is_enforced():
    """Re-fitted scalers are not interchangeable with the frozen ones."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _meta(tmp, "causal", "ev_scalers_causal_full_warm_e20_s0.joblib")
        assert_scaler_pairing(p, "x/ev_scalers_causal_full_warm_e20_s0.joblib")
        try:
            assert_scaler_pairing(p, "x/ev_scalers.joblib")
        except FeatureModeMismatch:
            return
        raise AssertionError("the frozen production scalers were accepted")


def test_scale_tripwire_catches_the_observed_ratio():
    """14.5 kW predicted against 56.9 realised is a ratio of 0.25."""
    msg = warn_if_prediction_scale_drifts(np.full(96, 14.5), 56.9)
    assert msg is not None and "ANOMALY" in msg


def test_scale_tripwire_is_quiet_when_healthy():
    """51.4 against 56.9 -- the refit -- must not warn."""
    assert warn_if_prediction_scale_drifts(np.full(96, 51.4), 56.9) is None


def test_scale_tripwire_survives_a_dead_site():
    """A zero realised mean must not divide by zero."""
    assert warn_if_prediction_scale_drifts(np.full(96, 10.0), 0.0) is None


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
