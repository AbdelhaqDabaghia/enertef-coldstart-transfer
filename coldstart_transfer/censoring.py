"""
censoring.py -- continual learning on a target censored by its own controller.

THE SETTING. The MPC commands the two charging groups and nothing else, so the
EV forecaster now adapts on data its own decisions shaped, while the PV
forecaster on the same site and the same pipeline adapts on data nothing
touched. What the controller does to the EV target is not a distribution shift;
it is CENSORING:

    y_t = max(y0_t + u_t, 0)

    y_t > 0             ->  y0_t = y_t - u_t        exactly recoverable
    y_t = 0 and u_t < 0 ->  y0_t <= -u_t            an upper bound only

`e26_censoring_probe.py` measured this on nine days of live control: 90.2 % of
slots commanded, 45 % of those censored, 1333 kWh of baseline demand rendered
unobservable. PV: zero on every count.

WHY THIS IS NOT THE PUBLISHED PROBLEM. Value-oriented and decision-focused
forecasting (Zhang et al., IEEE TSG 2024; Deriving Loss Function for
Value-oriented Renewable Energy Forecasting) align the training loss with the
downstream operation cost, and that is settled work. They operate on
RENEWABLE generation, where the control cannot move the target: you do not
curtail the sun into your own training set. Customer-baseline-load estimation
knows the counterfactual is unobservable, and Tobit regression handles censored
targets, but both are static estimators. What is left, and what this module
implements, is the intersection: a forecaster that RE-ADAPTS NIGHTLY on a
target its own policy censors, with a replay buffer that would otherwise
entrench the censoring by storing the contaminated data.

THREE PIECES, each addressing one way the contamination enters.

1. THE LOSS. A censored observation is not a target, it is a bound. Penalising
   a prediction for being below the bound is penalising it for being right.
   `censored_loss` applies the ordinary residual loss where the baseline is
   recoverable and a one-sided penalty where only the bound is known.

2. THE FEATURES. `lag_1`, `lag_4`, `roll_1h_mean` and the rest are built from
   the observed series, which is the CONTROLLED series. A model fitted on lags
   of controlled demand learns to predict the controller, not the demand: it
   becomes accurate in closed loop and blind in open loop, which is exactly the
   regime the counterfactual baseline lives in. `decontaminate` rebuilds them
   from the reconstructed baseline.

3. THE UNCERTAINTY, MADE EXPLICIT. Where the baseline is censored its
   reconstruction is a bound, not a value, so the rebuilt features are
   partially fabricated. Rather than impute silently -- the defect that filled
   a fifth of the lookback with the training median and reported nothing --
   `decontaminate` emits a companion `censored_frac` channel giving the
   fraction of each window that is a bound. The model is told where its own
   history is uncertain.
"""
from __future__ import annotations

import numpy as np

FLOOR_KW = 0.05   # below this the meter reads zero; see e26


# ---------------------------------------------------------------- recovery --
def reconstruct_baseline(y, u, floor=FLOOR_KW):
    """Recover the uncontrolled demand from the controlled observation.

    Args:
        y: observed (controlled) power, kW.
        u: the setpoint adjustment that was dispatched for the same slot, kW.
           Zero where nothing was commanded.

    Returns:
        y0        reconstructed baseline. Exact where recoverable; where
                  censored it holds the UPPER BOUND, which is the most that can
                  be said, and `censored` marks those entries so no caller can
                  mistake a bound for a measurement.
        censored  boolean mask, True where only a bound is known.
        bound     the upper bound itself, -u, valid where `censored`.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    censored = (y <= floor) & (u < 0)
    bound = np.where(censored, -u, np.inf)
    y0 = np.where(censored, bound, y - u)
    return y0, censored, bound


def censoring_report(y, u, floor=FLOOR_KW):
    """Summary of how contaminated a window is, for logging and for the paper."""
    y0, censored, bound = reconstruct_baseline(y, u, floor)
    commanded = int(np.sum(u != 0))
    n = len(y0)
    return dict(
        slots=n,
        commanded=commanded,
        commanded_frac=commanded / n if n else 0.0,
        censored=int(censored.sum()),
        censored_frac_of_commanded=(float(censored.sum() / commanded)
                                    if commanded else 0.0),
        hidden_kwh=float(np.sum(bound[censored]) * 0.25),
    )


# ------------------------------------------------------------------- loss --
def censored_loss(y_pred, y0, censored, bound, delta=0.5, weight_censored=1.0,
                  sigma=None):
    """Loss that treats a censored observation as a bound, not a target.

    Uncensored entries get the ordinary Huber residual loss, matching the
    deployed EV objective. Censored entries know only y0 <= bound, so:

      - a prediction at or below the bound is consistent with what was
        observed and costs nothing;
      - a prediction above the bound contradicts the observation and is
        penalised by how far it exceeds it.

    `sigma` selects the censored branch. None (default) uses the hinge, the
    sigma -> 0 limit: exact, free of a nuisance parameter, and it makes no
    distributional assumption. Given a positive sigma the branch instead uses
    the Gaussian Tobit term -log Phi((bound - pred)/sigma), which is smoother
    and is the right choice when the residual scale is known and worth
    modelling. We report the hinge as the default because the residual scale at
    this site is heteroscedastic across the diurnal cycle and a single sigma
    would be a fiction.

    Returns the mean loss over all entries, so censored and uncensored windows
    remain comparable.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    censored = np.asarray(censored, dtype=bool)

    out = np.zeros_like(y_pred)

    obs = ~censored
    if obs.any():
        r = np.abs(y_pred[obs] - y0[obs])
        out[obs] = np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))

    if censored.any():
        over = y_pred[censored] - bound[censored]
        if sigma is None:
            h = np.maximum(over, 0.0)
            out[censored] = weight_censored * np.where(
                h <= delta, 0.5 * h ** 2, delta * (h - 0.5 * delta))
        else:
            from math import sqrt
            z = -over / sigma
            # -log Phi(z), stable for z << 0 via the asymptotic tail
            phi = 0.5 * (1.0 + np.vectorize(_erf)(z / sqrt(2.0)))
            out[censored] = weight_censored * -np.log(np.clip(phi, 1e-12, None))

    return float(np.mean(out))


def _erf(x):
    import math
    return math.erf(x)


def censored_loss_grad(y_pred, y0, censored, bound, delta=0.5,
                       weight_censored=1.0):
    """d(loss)/d(y_pred) for the hinge variant, for gradient-based training.

    The censored branch has zero gradient below the bound: the model is free to
    predict anything there, which is correct, because anything there is
    consistent with what was observed.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    g = np.zeros_like(y_pred)

    obs = ~censored
    if obs.any():
        r = y_pred[obs] - y0[obs]
        g[obs] = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    if censored.any():
        over = y_pred[censored] - bound[censored]
        h = np.maximum(over, 0.0)
        g[censored] = weight_censored * np.where(
            h == 0.0, 0.0, np.where(h <= delta, h, delta))

    return g / len(y_pred)


# --------------------------------------------------------------- features --
LAGS = (1, 4, 96, 672)
ROLLS = {"roll_1h_mean": 4, "roll_6h_mean": 24, "roll_24h_mean": 96}


def decontaminate(y, u, floor=FLOOR_KW):
    """Rebuild the autoregressive features from the baseline, not the control.

    Returns a dict of feature name -> array, plus `censored_frac`: for each
    step, the fraction of the 24-hour window behind it that is a bound rather
    than a measurement. That channel is the point. The alternative is to impute
    the censored entries and say nothing, which is how a fifth of the lookback
    came to be filled with the training median while every metric reported
    success.

    Rolling means are causal by construction -- shifted by one step before the
    window is taken -- which is the convention the deployed serving path uses
    and the one the leaky training pipeline did not.
    """
    y0, censored, bound = reconstruct_baseline(y, u, floor)
    n = len(y0)
    out = {}

    for L in LAGS:
        v = np.full(n, np.nan)
        v[L:] = y0[:-L]
        out["lag_%d" % L] = v

    prev = np.concatenate([[np.nan], y0[:-1]])       # shift(1): causal
    for name, w in ROLLS.items():
        v = np.full(n, np.nan)
        cs = np.nancumsum(np.nan_to_num(prev))
        cnt = np.cumsum(~np.isnan(prev))
        for i in range(n):
            lo = max(0, i - w + 1)
            num = cs[i] - (cs[lo - 1] if lo > 0 else 0.0)
            den = cnt[i] - (cnt[lo - 1] if lo > 0 else 0)
            v[i] = num / den if den > 0 else np.nan
        out[name] = v

    cw = 96
    cens = censored.astype(float)
    frac = np.full(n, np.nan)
    cc = np.cumsum(cens)
    for i in range(n):
        lo = max(0, i - cw + 1)
        s = cc[i] - (cc[lo - 1] if lo > 0 else 0.0)
        frac[i] = s / (i - lo + 1)
    out["censored_frac"] = frac

    out["_y0"] = y0
    out["_censored"] = censored
    out["_bound"] = bound
    return out


# -------------------------------------------------------- replay hygiene --
def decontaminated_reservoir(buffer, y, u, size, rng, floor=FLOOR_KW):
    """Reservoir update that stores the BASELINE, never the controlled series.

    Bounded replay is the one mechanism that prevented seasonal forgetting in
    the multi-cycle study, but a buffer filled with controlled observations
    replays the controller's fingerprint and entrenches it. Censored entries
    are excluded outright: a bound is not a training pair, and admitting it as
    one would teach the model the floor rather than the demand.
    """
    y0, censored, _ = reconstruct_baseline(y, u, floor)
    keep = ~censored
    if not keep.any():
        return buffer, 0
    cand = y0[keep]
    if buffer is None or len(buffer) == 0:
        return cand[:size].copy(), int(min(len(cand), size))
    buf = np.asarray(buffer, dtype=float).copy()
    seen = len(buf)
    for v in cand:
        if len(buf) < size:
            buf = np.append(buf, v)
        else:
            j = rng.integers(0, seen + 1)
            if j < size:
                buf[j] = v
        seen += 1
    return buf, int(keep.sum())
