"""
validation_v2.py -- a promotion gate with an absolute floor.

Proposed replacement for enertef-cl-trainer/sagemaker-training/validation.py.
Not applied: that is a different repository and deploying is the operator's
call. Drop-in compatible -- validate_new_model keeps its signature and its
return shape, and the new checks are additive.

WHY THE CURRENT GATE IS NOT ENOUGH

It compares the candidate with the incumbent and allows a 5 % regression:

    if MAE(new) > MAE(baseline) * 1.05: reject

That protects against a bad update. It cannot protect against a bad lineage.
If the incumbent is already poor, a candidate that is merely no worse sails
through, and the gate reports PROMOTE while the model is useless in absolute
terms. That is not hypothetical here: the deployed EV model scores 1.389 nRMSE
against naive persistence at 0.997 (experiment e19), so it is beaten by a
one-step lag -- and every nightly candidate within 5 % of it would have been
promoted, forever, without the gate ever objecting.

Under causal features, two of the three adaptation mechanisms we tested lose to
persistence. A relative gate promotes them all.

THREE ADDITIONS

1. An absolute floor. A candidate that cannot beat naive persistence on the
   same holdout is not deployable, whatever the incumbent does. Persistence
   costs nothing to compute and needs no model.

2. A prediction-scale check. The deployed model predicted a mean of 14.5 kW
   against a realised 56.9 -- a quarter of reality, every day, visibly. MAE
   alone is a weak detector of that; a ratio of means is a strong one, and the
   controller downstream is far more sensitive to a systematic amplitude error
   than to symmetric noise.

3. A degenerate-baseline fix. The current Case 1 promotes ANY candidate when
   the baseline's metrics are degenerate:

       if np.isnan(baseline_mae) or baseline_mae == 0:
           return {"promote": True, ...}

   A missing or broken baseline is the moment to be more careful, not less --
   it usually means the holdout was empty or the incumbent failed to load.
   Here it falls through to the absolute checks instead, and promotes only if
   the candidate can stand on its own.
"""
from __future__ import annotations
import numpy as np


def compute_metrics(y_true, y_pred):
    """Unchanged from validation.py, so callers keep working."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "nmae": float("nan")}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.mean(np.abs(y_true))
    nmae = float(mae / denom) if denom > 1e-6 else float("nan")
    return {"mae": mae, "rmse": rmse, "nmae": nmae}


def persistence_forecast(ground_truth):
    """The one-step lag. No model, no features, no training -- which is the
    point: anything that cannot beat this is not earning its deployment."""
    y = np.asarray(ground_truth, dtype=np.float64)
    if len(y) == 0:
        return y
    return np.concatenate([[y[0]], y[:-1]])


def _reject(reason, bm, nm, extra=None):
    out = {"promote": False, "baseline_metrics": bm, "new_metrics": nm,
           "reason": reason}
    out.update(extra or {})
    return out


def validate_new_model(baseline_predictions, new_predictions, ground_truth,
                       tolerance=1.05, min_improvement=None,
                       require_beat_persistence=True,
                       scale_bounds=(0.5, 2.0)):
    """Decide whether a candidate may be promoted.

    Additions over validation.py, both defaulting ON:
      require_beat_persistence  reject a candidate that a one-step lag beats
      scale_bounds              reject when mean(pred)/mean(true) falls outside
                                this band; None disables

    Returns the same dict as before, plus 'persistence_metrics' and
    'scale_ratio' so the caller can log what it was judged against.
    """
    gt = np.asarray(ground_truth, dtype=np.float64)
    bm = compute_metrics(gt, baseline_predictions)
    nm = compute_metrics(gt, new_predictions)
    pm = compute_metrics(gt, persistence_forecast(gt))
    extra = {"persistence_metrics": pm}

    # --- the candidate must be a valid model at all -------------------------
    if len(gt) == 0:
        return _reject("Empty holdout: nothing to validate. Refusing to "
                       "promote on no evidence.", bm, nm, extra)

    new_mae = nm["mae"]
    if np.isnan(new_mae) or np.isinf(new_mae):
        return _reject("New model produced invalid predictions (MAE = %s)."
                       % new_mae, bm, nm, extra)

    # --- absolute floor: beat a one-step lag --------------------------------
    # Checked BEFORE the relative comparison, because a candidate that loses to
    # persistence should not be promotable however bad the incumbent is.
    if require_beat_persistence and not np.isnan(pm["mae"]):
        if new_mae >= pm["mae"]:
            return _reject(
                "BELOW ABSOLUTE FLOOR: new MAE %.4f does not beat naive "
                "persistence %.4f on the same holdout. A model that a one-step "
                "lag outperforms is not deployable regardless of how it "
                "compares with the incumbent." % (new_mae, pm["mae"]),
                bm, nm, extra)

    # --- prediction scale ---------------------------------------------------
    ratio = None
    true_mean = float(np.mean(gt))
    if scale_bounds is not None and abs(true_mean) > 1e-6:
        ratio = float(np.mean(np.asarray(new_predictions, float)) / true_mean)
        extra["scale_ratio"] = ratio
        lo, hi = scale_bounds
        if not (lo <= ratio <= hi):
            return _reject(
                "PREDICTION SCALE: mean forecast is %.2fx the realised mean "
                "(allowed %.2f-%.2f). This is the signature of a train/serve "
                "feature mismatch, and a systematic amplitude error costs a "
                "scheduler far more than symmetric noise of the same MAE."
                % (ratio, lo, hi), bm, nm, extra)

    baseline_mae = bm["mae"]

    # --- degenerate baseline ------------------------------------------------
    # validation.py promoted unconditionally here. A missing baseline usually
    # means an empty holdout or an incumbent that failed to load; by this point
    # the candidate has already cleared the absolute checks, so promote on
    # those alone and say so.
    if np.isnan(baseline_mae) or baseline_mae == 0:
        return {"promote": True, "baseline_metrics": bm, "new_metrics": nm,
                "reason": ("Baseline metrics degenerate; promoted on the "
                           "absolute checks alone (MAE %.4f vs persistence "
                           "%.4f). Investigate why the baseline is missing."
                           % (new_mae, pm["mae"])), **extra}

    # --- relative comparison, as before ------------------------------------
    max_allowed = baseline_mae * tolerance
    if new_mae > max_allowed:
        return _reject(
            "REGRESSION: new MAE %.4f exceeds baseline %.4f * tolerance %s = "
            "%.4f. Keeping baseline." % (new_mae, baseline_mae, tolerance,
                                         max_allowed), bm, nm, extra)

    if min_improvement is not None:
        required = baseline_mae * (1.0 - min_improvement)
        if new_mae > required:
            return _reject(
                "INSUFFICIENT IMPROVEMENT: new MAE %.4f not below required "
                "%.4f (%.0f%% improvement required). Keeping baseline."
                % (new_mae, required, min_improvement * 100), bm, nm, extra)

    delta = 100 * (baseline_mae - new_mae) / baseline_mae
    return {"promote": True, "baseline_metrics": bm, "new_metrics": nm,
            "reason": ("PROMOTE: new MAE %.4f vs baseline %.4f (delta %+.2f%%), "
                       "persistence %.4f%s."
                       % (new_mae, baseline_mae, delta, pm["mae"],
                          "" if ratio is None else ", scale %.2fx" % ratio)),
            **extra}
