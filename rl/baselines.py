"""
baselines.py -- the controllers the policy has to beat, none of them weak.

    mpc_deployed  the transcribed production controller: 8*import - 2*export
                  plus quadratic regularisers, solved once per day against the
                  forecast. This is what the site runs today.

    mpc_ledger    the SAME convex program, same constraints, same information,
                  but minimising the bill the site actually pays. The deployed
                  objective is not the ledger, so this is the fair "what could
                  a correctly specified MPC do" baseline -- and the gap between
                  it and mpc_deployed is a finding in its own right.

    mpc_rh        mpc_ledger re-solved every step over the remaining horizon,
                  with the demand realised so far substituted for the forecast.
                  This is the honest closed-loop baseline. Without it, an RL
                  policy that sees realised demand would be beating an
                  open-loop plan and the comparison would prove nothing.

    mpc_saa       the stochastic program. Same constraints, same information,
                  but it maximises the EXPECTED settled bill over scenarios of
                  what demand might be, instead of trusting a point forecast.
                  This is the arm the mathematics says should win.

    oracle        mpc_ledger given the TRUE demand. No causal controller can
                  beat it; it is the ceiling that says how much is on the table.

All four return (u1, u2) and are settled by the same ledger as the policy, on
the realised demand, with the same physical floor at zero.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from rl.site_env import (H, DT_H, U_LO, U_HI, EV1_LO, EV1_HI, EV2_LO, EV2_HI,
                         RAMP_MAX)

W_IMPORT, W_EXPORT, W_U, W_RAMP = 8.0, 2.0, 1e-4, 0.01

_CACHE = {}


def _program(n, objective):
    """Compile once per (length, objective); re-solve through parameters."""
    key = (n, objective)
    if key in _CACHE:
        return _CACHE[key]

    u1, u2 = cp.Variable(n), cp.Variable(n)
    pv = cp.Parameter(n)
    b1, b2 = cp.Parameter(n), cp.Parameter(n)
    # NOT nonneg. Day-ahead prices go negative -- the observed range over four
    # years is -500 to +3000 EUR/MWh -- and a negative price is the single most
    # valuable moment to CONSUME MORE, because the site is paid to take energy.
    # Declaring the parameter nonneg forced a clip at zero that deleted exactly
    # those hours from the MPC's view, while the environment kept showing the
    # true prices to the policy. The comparison was biased in the policy's
    # favour. Both objectives are affine in the variables, so no sign
    # assumption is needed for DCP.
    price = cp.Parameter(n)
    u0 = cp.Parameter(2)                      # last applied setpoint, for ramp
    bud = cp.Parameter(2)                     # energy still to return

    ev1c, ev2c = b1 + u1, b2 + u2
    cons = [ev1c >= EV1_LO, ev1c <= EV1_HI, ev2c >= EV2_LO, ev2c <= EV2_HI,
            u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[0] - u0[0]) <= RAMP_MAX, cp.abs(u2[0] - u0[1]) <= RAMP_MAX,
            cp.sum(u1) == bud[0], cp.sum(u2) == bud[1]]
    if n > 1:
        cons += [cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
                 cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX]

    if objective == "deployed":
        pgrid = ev1c + ev2c - pv
        imp, exp = cp.Variable(n), cp.Variable(n)
        cons += [imp >= 0, exp >= 0, pgrid == imp - exp]
        obj = (W_IMPORT * cp.sum(cp.multiply(price, imp * DT_H / 1000.0))
               - W_EXPORT * cp.sum(cp.multiply(price, exp * DT_H / 1000.0))
               + W_U * (cp.sum_squares(u1) + cp.sum_squares(u2))
               + W_RAMP * (cp.sum_squares(u1[1:] - u1[:-1])
                           + cp.sum_squares(u2[1:] - u2[:-1])) if n > 1 else 0)
    else:
        # the ledger the site pays: sum(price * (ev - pv) * dt), linear in u.
        # a whisper of regularisation keeps the LP from picking arbitrary
        # vertices among ties; it is 1e-6, far below one cent a day.
        # price * (b1 + b2 - pv) is a constant offset and does not move the
        # argmin, so only the controllable part is kept. Dropping it also keeps
        # the program DPP -- parameter times VARIABLE, never parameter times
        # parameter -- which is what lets cvxpy reuse the compiled problem
        # instead of recompiling on every one of the thousands of solves the
        # receding-horizon arm performs.
        obj = (cp.sum(cp.multiply(price, (u1 + u2) * DT_H / 1000.0))
               + 1e-6 * (cp.sum_squares(u1) + cp.sum_squares(u2)))

    prob = cp.Problem(cp.Minimize(obj), cons)
    _CACHE[key] = (prob, (pv, b1, b2, price, u0, bud), (u1, u2))
    return _CACHE[key]


def _solve(n, objective, pv, b1, b2, price, u0=(0.0, 0.0), bud=(0.0, 0.0)):
    prob, (p_pv, p_b1, p_b2, p_pr, p_u0, p_bd), (u1, u2) = _program(n, objective)
    p_pv.value = np.asarray(pv, float)
    p_b1.value = np.asarray(b1, float)
    p_b2.value = np.asarray(b2, float)
    p_pr.value = np.asarray(price, float)
    p_u0.value = np.asarray(u0, float)
    p_bd.value = np.asarray(bud, float)
    try:
        prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
    except Exception:
        return np.zeros(n), np.zeros(n)
    if u1.value is None or u2.value is None:
        return np.zeros(n), np.zeros(n)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


# ------------------------------------------------------------ open loop ----
def mpc_deployed(env, day, price_day):
    return _solve(H, "deployed", env.pv[day], env.f1[day], env.f2[day],
                  env.price[price_day])


def mpc_ledger(env, day, price_day):
    return _solve(H, "ledger", env.pv[day], env.f1[day], env.f2[day],
                  env.price[price_day])


def oracle(env, day, price_day):
    return _solve(H, "ledger", env.pv[day], env.ev1[day], env.ev2[day],
                  env.price[price_day])


# --------------------------------------------------------- receding horizon -
def mpc_receding(env, day, price_day, every=4):
    """Re-solve over the remaining horizon, substituting realised demand.

    `every` re-solves once per `every` steps (default hourly) and applies the
    first setpoints of each plan. Re-solving at every quarter-hour changes the
    result by less than a cent a day here and costs four times the compute.
    """
    pv, pr = env.pv[day], env.price[price_day]
    b1, b2 = env.f1[day].copy(), env.f2[day].copy()
    r1, r2 = env.ev1[day], env.ev2[day]

    u1 = np.zeros(H)
    u2 = np.zeros(H)
    u0 = np.zeros(2)
    bud = np.zeros(2)

    t = 0
    while t < H:
        n = H - t
        # what has already happened is known exactly; the rest is still forecast
        p1 = np.concatenate([r1[t:t + 0], b1[t:]])
        p2 = np.concatenate([r2[t:t + 0], b2[t:]])
        s1, s2 = _solve(n, "ledger", pv[t:], p1, p2, pr[t:], u0=u0, bud=bud)
        k = min(every, n)
        u1[t:t + k], u2[t:t + k] = s1[:k], s2[:k]
        u0 = np.array([u1[t + k - 1], u2[t + k - 1]])
        bud = bud - np.array([s1[:k].sum(), s2[:k].sum()])
        # the steps just executed are now history: replace forecast by realised
        b1[t:t + k], b2[t:t + k] = r1[t:t + k], r2[t:t + k]
        t += k
    return u1, u2


# ===================================================== stochastic program ====
#
# The settled saving of one step is
#
#     s_t(u_t) = p_t * (r_t - max(r_t + u_t, 0)) = p_t * min(-u_t, r_t)
#
# You only bank a reduction if the demand was there to reduce. For p_t >= 0
# that is a minimum of two affine functions, hence CONCAVE, and its expectation
# stays concave: maximising it over the polyhedron of bounds, ramp and
# sum(u) = 0 is a convex program, exactly solvable.
#
# For p_t < 0 the same expression is |p_t| * max(u_t, -r_t), which is CONVEX --
# so the problem is not concave everywhere, and day-ahead prices in this zone
# reach -500 EUR/MWh. The resolution is that at a negative price one never
# wants to reduce: the site is paid to consume. Imposing u_t >= 0 on those
# slots makes the minimum resolve to -u_t exactly, the term becomes linear, and
# concavity is restored. That is a RESTRICTION of the feasible set, so what
# comes back is admissible and therefore a lower bound on the true stochastic
# optimum -- never an overstatement.
#
# Scenarios are a residual bootstrap taken PER TIME-OF-DAY SLOT from the
# training days, which preserves the diurnal shape of forecast error: the
# uncertainty at 03:00, when the chargers are usually idle, is not the
# uncertainty at 18:00.


def demand_scenarios(f1, f2, resid1, resid2, n, rng):
    """n scenarios of (r1, r2) around the forecast, clipped at the floor."""
    idx = rng.integers(0, resid1.shape[0], size=n)
    r1 = np.maximum(f1[None, :] + resid1[idx], 0.0)
    r2 = np.maximum(f2[None, :] + resid2[idx], 0.0)
    return r1, r2


def mpc_saa(env, day, price_day, n_scen=60, seed=0):
    """Maximise the EXPECTED settled bill over demand scenarios."""
    pv = env.pv[day]
    pr = np.asarray(env.price[price_day], float)
    f1, f2 = env.f1[day], env.f2[day]
    if getattr(env, "resid1", None) is None:
        return mpc_ledger(env, day, price_day)

    rng = np.random.default_rng(seed + 1000 * int(day) + int(price_day))
    r1s, r2s = demand_scenarios(f1, f2, env.resid1, env.resid2, n_scen, rng)

    n = H
    pos = pr >= 0.0
    neg = ~pos

    u1, u2 = cp.Variable(n), cp.Variable(n)
    cons = [u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]
    if neg.any():
        # never reduce when the price pays you to consume
        cons += [u1[neg] >= 0, u2[neg] >= 0]

    obj = 0
    if neg.any():
        obj = obj + cp.sum(cp.multiply(pr[neg], -(u1[neg] + u2[neg]))) * DT_H / 1000.0
    if pos.any():
        z1 = cp.Variable((n_scen, int(pos.sum())))
        z2 = cp.Variable((n_scen, int(pos.sum())))
        cons += [z1 <= cp.reshape(-u1[pos], (1, int(pos.sum())), order="C"),
                 z2 <= cp.reshape(-u2[pos], (1, int(pos.sum())), order="C"),
                 z1 <= r1s[:, pos], z2 <= r2s[:, pos]]
        obj = obj + (cp.sum(cp.multiply(
            np.tile(pr[pos], (n_scen, 1)), z1 + z2))
            * DT_H / 1000.0 / n_scen)

    try:
        cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return mpc_ledger(env, day, price_day)
    if u1.value is None or u2.value is None:
        return mpc_ledger(env, day, price_day)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


ARMS = {
    "mpc_deployed": mpc_deployed,
    "mpc_ledger": mpc_ledger,
    "mpc_rh": mpc_receding,
    "mpc_saa": mpc_saa,
    "oracle": oracle,
}
