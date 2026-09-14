"""
test_causality.py -- does a feature available at time t depend on y_t?

THE TEST. Perturb ev_kw at one index t and rebuild the features. Every feature
at rows <= t must be unchanged; only rows > t may move. A feature at row t that
reacts to y_t is, by construction, unavailable at inference and leaks the target.

The tests are written to PASS on causal=True and FAIL on causal=False, and the
failing case is asserted explicitly rather than skipped: the leak in the default
pipeline is a measured property of this repository, not an accident we hide.

Run:  python -m pytest tests/test_causality.py -v
      python tests/test_causality.py        (no pytest needed)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from coldstart_transfer.features import (engineer_features, SEQ_FEATURES,
                                         NEXT_EXO, WEATHER_VARS)
from coldstart_transfer.windowing import LOOKBACK

N = 900
T = 700          # the index we perturb; > LOOKBACK so windows exist either side
DERIVED = ["lag_1", "lag_4", "lag_96", "lag_672",
           "roll_1h_mean", "roll_6h_mean", "roll_24h_mean"]


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=N, freq="15min", tz="UTC")
    d = {"timestamp": ts.astype(str),
         "ev_kw": rng.gamma(2.0, 20.0, N)}
    for w in WEATHER_VARS:
        d[w] = rng.normal(0, 1, N)
    return pd.DataFrame(d)


def _rebuild(causal):
    """Return (base, perturbed) feature frames differing only in ev_kw[T]."""
    a = _frame()
    b = a.copy()
    b.loc[T, "ev_kw"] = a.loc[T, "ev_kw"] + 1000.0     # unmistakable perturbation
    return (engineer_features(a, causal=causal),
            engineer_features(b, causal=causal))


def _first_touched_row(fa, fb, col):
    """Lowest row index at which `col` differs, or None."""
    d = np.abs(fa[col].to_numpy(float) - fb[col].to_numpy(float))
    idx = np.flatnonzero(d > 1e-9)
    return int(idx[0]) if len(idx) else None


# ----------------------------------------------------------------- causal path

def test_causal_features_do_not_see_the_present():
    """Under causal=True no derived feature at row <= T reacts to ev[T]."""
    fa, fb = _rebuild(causal=True)
    offenders = {}
    for c in DERIVED:
        first = _first_touched_row(fa, fb, c)
        if first is not None and first <= T:
            offenders[c] = first
    assert not offenders, (
        "causal=True still leaks: these features change at or before the "
        f"perturbed row {T}: {offenders}")


def test_causal_features_do_react_afterwards():
    """A shifted feature must still carry the information -- one step later.
    Guards against 'fixing' the leak by destroying the feature."""
    fa, fb = _rebuild(causal=True)
    first = _first_touched_row(fa, fb, "roll_1h_mean")
    assert first == T + 1, (
        f"roll_1h_mean should first move at {T + 1} (one step after the "
        f"perturbation); it moved at {first}")


def test_next_exo_vector_is_clean():
    """The real exposure: windowing hands X_next = NEXT_EXO at the PREDICTED
    step. Under causal=True that vector must not depend on the target it
    accompanies."""
    fa, fb = _rebuild(causal=True)
    dirty = [c for c in NEXT_EXO
             if abs(float(fa[c].iloc[T]) - float(fb[c].iloc[T])) > 1e-9]
    assert not dirty, (
        f"X_next at the predicted step depends on y_t through {dirty}")


def test_time_encodings_are_untouched():
    """Sanity: calendar features cannot move at all."""
    fa, fb = _rebuild(causal=True)
    for c in ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
              "is_weekend"] + WEATHER_VARS:
        assert _first_touched_row(fa, fb, c) is None, f"{c} moved; it must not"


# ------------------------------------------------- the historical path, pinned

def test_default_pipeline_leaks_and_we_say_so():
    """The default (causal=False) DOES leak. Asserted, not skipped, so that the
    day someone changes the default this test tells them what they changed."""
    fa, fb = _rebuild(causal=False)
    leaking = [c for c in ["roll_1h_mean", "roll_6h_mean", "roll_24h_mean"]
               if abs(float(fa[c].iloc[T]) - float(fb[c].iloc[T])) > 1e-9]
    assert leaking == ["roll_1h_mean", "roll_6h_mean", "roll_24h_mean"], (
        "the historical convention was expected to leak through all three "
        f"rolling means; observed {leaking}. If this now passes cleanly the "
        "default has changed and every archived result needs re-labelling.")


def test_leak_magnitude_is_what_theory_says():
    """roll_1h_mean is a 4-point mean, so a +1000 perturbation must move it by
    exactly 250. Confirms the leak is the arithmetic one we claim, not a
    coincidence of indices."""
    fa, fb = _rebuild(causal=False)
    moved = float(fb["roll_1h_mean"].iloc[T]) - float(fa["roll_1h_mean"].iloc[T])
    assert abs(moved - 250.0) < 1e-6, f"expected +250.0, got {moved}"


def test_lags_were_never_the_problem():
    """Lags are shifts and are causal under BOTH settings. Stated so the
    write-up does not over-attribute the leak."""
    for causal in (False, True):
        fa, fb = _rebuild(causal=causal)
        for c in ["lag_1", "lag_4", "lag_96", "lag_672"]:
            first = _first_touched_row(fa, fb, c)
            assert first is None or first > T, (
                f"{c} leaked with causal={causal} at row {first}")


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
    raise SystemExit(1 if bad else 0)
