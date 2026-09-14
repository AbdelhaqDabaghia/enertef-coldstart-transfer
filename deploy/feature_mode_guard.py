"""
feature_mode_guard.py -- refuse to serve a model against the wrong features.

The defect this repository spent a day diagnosing was a model fitted on one
rolling-mean convention and served another. It was invisible for months because
nothing in the stack asserted that the two agreed: the weights carried no
statement of what they expected, and the serving code carried no check.

The retrained artefacts now declare their convention in
ev_<tag>_metadata.json ("feature_mode": "causal"). This module turns that
declaration into a runtime assertion. Call it once at service start, before the
first forecast.

It is deliberately cheap and deliberately fatal. A service that refuses to start
is a page at 09:00; a service that quietly serves a mis-specified model is a
quarter of the true demand on a dashboard for a year, and a KPI nobody can
reproduce.

USAGE (in realtime_runner, after loading the model)

    from feature_mode_guard import assert_feature_mode
    assert_feature_mode(metadata_path, builder_mode="serve")

where builder_mode is what THIS service actually computes. Today
ev_features_builder computes 'serve'; once it is corrected to the causal
convention, change the argument at the same time and this guard will confirm
the pair rather than block it.
"""
from __future__ import annotations
import json
import os
import numpy as np


class FeatureModeMismatch(RuntimeError):
    """Raised when the served features are not what the model was fitted on."""


def assert_feature_mode(metadata_path: str, builder_mode: str) -> dict:
    """Load a model's metadata and refuse to proceed on a mismatch.

    metadata_path  ev_<tag>_metadata.json written by retrain_causal.py
    builder_mode   what the serving feature builder computes: one of
                   'train', 'causal', 'serve'

    Returns the metadata on success, so the caller can log the provenance it
    just validated.
    """
    if not os.path.exists(metadata_path):
        raise FeatureModeMismatch(
            "no metadata at %s. A model without a declared feature convention "
            "cannot be validated, and serving it is how the original defect "
            "went unnoticed. Retrain with scripts/retrain_causal.py, which "
            "writes it, or add the declaration by hand after confirming what "
            "the weights were fitted on." % metadata_path)

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    want = meta.get("feature_mode")
    if want is None:
        raise FeatureModeMismatch(
            "%s declares no feature_mode." % metadata_path)

    if want != builder_mode:
        raise FeatureModeMismatch(
            "FEATURE CONVENTION MISMATCH -- refusing to serve.\n"
            "  model expects : %r  (%s)\n"
            "  builder serves: %r\n"
            "  A model fitted on one rolling-mean convention and served "
            "another under-predicted this site by a factor of 3.9 while every "
            "dashboard reported success. Either retrain for %r or correct the "
            "feature builder." % (want, metadata_path, builder_mode, builder_mode))
    return meta


def assert_scaler_pairing(metadata_path: str, scaler_path: str) -> None:
    """The scalers were re-fitted for this convention and are not
    interchangeable with the frozen production ones. Confirm the file actually
    loaded is the one the weights were trained with."""
    with open(metadata_path, encoding="utf-8") as f:
        expected = json.load(f).get("scalers")
    got = os.path.basename(scaler_path)
    if expected and expected != got:
        raise FeatureModeMismatch(
            "scaler mismatch: the model was trained with %r but %r was loaded. "
            "Re-fitted scalers and weights must be deployed together."
            % (expected, got))


def warn_if_prediction_scale_drifts(pred_kw, recent_true_mean_kw,
                                    lo: float = 0.5, hi: float = 2.0):
    """A cheap online tripwire for the same class of fault.

    The deployed model predicted a mean of 14.5 kW against a true 56.9 --
    a ratio of 0.25 -- every day, visibly, for as long as it ran. Any check on
    the ratio of predicted to recent realised mean would have caught it.

    Returns None when healthy, otherwise a message to log loudly. Deliberately
    not fatal: a genuine regime change can move this legitimately, and refusing
    to forecast would be worse than forecasting badly with a warning.
    """
    pm = float(np.mean(pred_kw))
    if recent_true_mean_kw <= 0:
        return None
    ratio = pm / float(recent_true_mean_kw)
    if lo <= ratio <= hi:
        return None
    return ("PREDICTION SCALE ANOMALY: forecast mean %.1f kW is %.2fx the "
            "recent realised mean %.1f kW (expected %.2f-%.2f). This is the "
            "signature of a train/serve feature mismatch; check the model's "
            "feature_mode against what the builder computes."
            % (pm, ratio, float(recent_true_mean_kw), lo, hi))
