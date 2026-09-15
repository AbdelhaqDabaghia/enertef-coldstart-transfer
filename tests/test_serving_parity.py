"""
test_serving_parity.py -- do training and serving compute the same features?

This test loads the REAL production builder (svc1-runner/ev_features_builder.py)
and compares what it produces, step by step, against what the training pipeline
produces for the same series. If the two disagree, the deployed model is being
fed something it was never fitted on, and every accuracy figure taken from
training overstates what the site receives.

They currently disagree, by a lot, and this test says so rather than skipping.
e19 measures the consequence: the deployed forecast under-predicts by a factor
of 3.9 (mean 14.5 kW against a true 56.9 kW).

WHY IT HAPPENS
compute_lag_features takes the window [idx-w+1, idx+1), inclusive of idx.
realtime_runner calls it through build_ev_next_input BEFORE the step at idx is
predicted, on an ev_series_full whose forecast slots are still zero. So the
service sums w-1 real values and divides by w.

    train   roll_w[t] = mean(y[t-w+1 .. t])        fitted on this
    serve   roll_w[t] = sum(y[t-w .. t-1]) / w     served this

Point PROD_BUILDER at the deployed file (env var overrides the default) and run
this in CI. It is the guard that stops the two pipelines drifting apart again.

    python tests/test_serving_parity.py
"""
from __future__ import annotations
import os
import sys
import importlib.util
import numpy as np
import pandas as pd

from coldstart_transfer.features import engineer_features, WEATHER_VARS

PROD_BUILDER = os.environ.get(
    "PROD_BUILDER", r"C:\dev\svc1-runner\ev_features_builder.py")
ROLLS = [("roll_1h_mean", 4), ("roll_6h_mean", 24), ("roll_24h_mean", 96)]
N, IDX = 900, 800


def _load_prod():
    """Import the deployed builder by path. It imports only numpy."""
    if not os.path.exists(PROD_BUILDER):
        return None
    spec = importlib.util.spec_from_file_location("prod_builder", PROD_BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _series(seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 20.0, N).astype(float)


def _frame(ev):
    ts = pd.date_range("2026-01-01", periods=len(ev), freq="15min", tz="UTC")
    d = {"timestamp": ts.astype(str), "ev_kw": ev}
    rng = np.random.default_rng(1)
    for w in WEATHER_VARS:
        d[w] = rng.normal(0, 1, len(ev))
    return pd.DataFrame(d)


def test_our_serve_mode_reproduces_production():
    """Our mode='serve' must match the deployed arithmetic exactly. If this
    fails, every claim we make about what the site receives is unfounded."""
    prod = _load_prod()
    if prod is None:
        print("  SKIP  production builder not found at %s" % PROD_BUILDER)
        return
    ev = _series()

    # what production feeds at prediction time: the slot at IDX is still zero
    served = ev.copy()
    served[IDX] = 0.0
    got = prod.compute_lag_features(served, IDX)

    ours = engineer_features(_frame(ev), mode="serve").iloc[IDX]
    for col, w in ROLLS:
        a, b = float(got[col]), float(ours[col])
        assert abs(a - b) < 1e-6, (
            "mode='serve' does not reproduce production for %s: "
            "production %.6f, ours %.6f" % (col, a, b))
    for lag in ["lag_1", "lag_4", "lag_96", "lag_672"]:
        a, b = float(got[lag]), float(ours[lag])
        assert abs(a - b) < 1e-6, "%s differs: %.6f vs %.6f" % (lag, a, b)


# --------------------------------------------------------------- the fixture
# The deployed builder is in a private repository, so CI cannot load it. The
# fixture records what it produced -- numbers only, no production logic -- so
# the parity claim is still checked on every push instead of silently skipped.

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "serving_parity.json")


def _fixture():
    import json
    if not os.path.exists(FIXTURE):
        return None
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_serve_mode_matches_the_captured_production_behaviour():
    """Runs everywhere, including CI. Proves OUR side has not drifted."""
    fx = _fixture()
    assert fx is not None, (
        "no fixture at %s. Regenerate with scripts/make_parity_fixture.py on a "
        "machine that has the deployed builder." % FIXTURE)

    rng = np.random.default_rng(fx["series_seed"])
    ev = rng.gamma(2.0, 20.0, fx["series_n"]).astype(float)
    ours_all = engineer_features(_frame(ev), mode="serve")

    for case in fx["cases"]:
        row = ours_all.iloc[case["idx"]]
        for name, expected in case["features"].items():
            got = float(row[name])
            assert abs(got - expected) < 1e-6, (
                "idx %d, %s: we now compute %.6f, the deployed builder produced "
                "%.6f when the fixture was captured (%s). Either our serve mode "
                "changed, or the fixture is stale -- regenerate it against the "
                "live builder to tell which."
                % (case["idx"], name, got, expected, fx["generated_utc"]))


def test_fixture_still_describes_the_live_builder():
    """Runs only where the real builder exists. Proves THEIR side has not
    drifted. A committed fixture is a snapshot, and this is what stops the
    snapshot quietly going stale -- the same failure class the project exists
    to fix."""
    fx = _fixture()
    prod_path = PROD_BUILDER
    if fx is None or not os.path.exists(prod_path):
        print("  SKIP  needs both the fixture and the deployed builder")
        return

    import hashlib
    sha = hashlib.sha256(open(prod_path, "rb").read()).hexdigest()
    if sha == fx["builder_sha256"]:
        return

    # The file changed. That is allowed -- but the captured behaviour must not.
    prod = _load_prod()
    rng = np.random.default_rng(fx["series_seed"])
    ev = rng.gamma(2.0, 20.0, fx["series_n"]).astype(float)
    for case in fx["cases"]:
        served = ev.copy()
        served[case["idx"]] = 0.0
        got = prod.compute_lag_features(served, case["idx"])
        for name, expected in case["features"].items():
            assert abs(float(got[name]) - expected) < 1e-6, (
                "THE DEPLOYED BUILDER HAS CHANGED BEHAVIOUR. %s at idx %d is now "
                "%.6f, was %.6f when the fixture was captured on %s. Every "
                "measurement in this repository about what the site receives "
                "describes the old builder. Re-measure, then regenerate the "
                "fixture." % (name, case["idx"], float(got[name]), expected,
                              fx["generated_utc"]))
        # behaviour intact, only the file text moved -- refresh the hash
    print("  NOTE  builder file changed (%s -> %s) but the captured behaviour "
          "is unchanged; refresh the fixture hash when convenient"
          % (fx["builder_sha256"][:12], sha[:12]))


def test_training_and_serving_disagree_today():
    """The defect, pinned. Asserted rather than skipped so that the day someone
    fixes the builder, this test tells them what they changed."""
    ev = _series()
    tr = engineer_features(_frame(ev), mode="train").iloc[IDX]
    sv = engineer_features(_frame(ev), mode="serve").iloc[IDX]
    gaps = {c: float(tr[c]) - float(sv[c]) for c, _ in ROLLS}
    assert all(g > 0 for g in gaps.values()), (
        "serving is expected to under-state every rolling mean; got %s" % gaps)
    # the 1 h window is the one that matters: it should be short by about
    # y_t/4 plus the re-weighting of the three known values
    assert gaps["roll_1h_mean"] > gaps["roll_6h_mean"] > gaps["roll_24h_mean"], (
        "the deficit must shrink with window length; got %s" % gaps)


def test_deficit_is_the_arithmetic_we_claim():
    """serve == causal - y[t-w]/w, exactly. Confirms the mechanism rather than
    inferring it from a degraded score.

    Note the windows differ: production's window is [t-w+1, t] with the last
    slot zero, while `causal` averages [t-w, t-1]. So the relation is a missing
    term, not a rescaling."""
    ev = _series()
    ca = engineer_features(_frame(ev), mode="causal").iloc[IDX]
    sv = engineer_features(_frame(ev), mode="serve").iloc[IDX]
    for col, w in ROLLS:
        expect = float(ca[col]) - ev[IDX - w] / w
        assert abs(float(sv[col]) - expect) < 1e-6, (
            "%s: serve %.6f != causal - y[t-w]/w = %.6f"
            % (col, sv[col], expect))


def test_no_serving_only_fix_can_match_training():
    """The acceptance criterion, stated as an impossibility.

    Dividing by the count of KNOWN values (w-1 rather than w) repairs the
    arithmetic but produces a (w-1)-point causal mean -- a third convention,
    matching neither `train` nor `causal`. There is therefore no serving-side
    patch that makes the deployed model correct: one causal convention has to
    be defined, implemented in both places, and the model retrained."""
    ev = _series()
    tr = engineer_features(_frame(ev), mode="train").iloc[IDX]
    ca = engineer_features(_frame(ev), mode="causal").iloc[IDX]
    for col, w in ROLLS:
        divide_by_known = float(np.mean(ev[IDX - (w - 1):IDX]))
        assert abs(divide_by_known - float(tr[col])) > 1e-6, (
            "%s: a serving-only fix unexpectedly reproduced training" % col)
        assert abs(divide_by_known - float(ca[col])) > 1e-9, (
            "%s: a serving-only fix unexpectedly reproduced the causal "
            "convention" % col)


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
