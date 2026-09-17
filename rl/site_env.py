"""
site_env.py -- the charging site as a closed-loop control problem.

One episode is one day: 96 quarter-hours, two metering points, a day-ahead
price curve that is KNOWN at the start, a PV profile, and an EV demand that the
controller only ever sees as a forecast until each step has passed.

Why this is not the same problem the deployed MPC solves, and why that matters:

1. THE LEDGER IS LINEAR. `bill()` computes import cost minus export revenue at
   the same price, and import minus export is just the grid position, so the
   settled bill is sum(price * (ev - pv) * dt). Savings against an unmanaged
   site are therefore -sum(price * u). The deployed MPC does NOT minimise this:
   it minimises 8*import - 2*export plus quadratic regularisers. It is optimal
   for its own objective, not for the bill the site actually pays. That gap is
   real money and it is available to any controller that targets the ledger
   directly.

2. THE DEPLOYED MPC IS OPEN LOOP. It solves once for the whole day against a
   point forecast. A policy acting step by step sees the demand that has
   actually arrived so far, and can correct. That is genuine information, not a
   modelling trick -- but it is also why `baselines.py` includes a
   receding-horizon MPC. Beating an open-loop plan with a closed-loop policy
   and calling it an RL win would be dishonest; the honest baseline re-solves.

FEASIBILITY IS ENFORCED, NOT LEARNED. Every action is clipped into the exact
admissible set of the deployed controller -- the same +/-200 kW bound, the same
80 kW ramp, the same per-site charger ceiling, the same sum(u) = 0 energy
conservation. The agent cannot win by cheating a constraint, and the resulting
setpoints are deployable as they stand. The closure clip is ramp-aware: it
reserves exactly the capacity still needed to return the running budget to
zero, computed from the true ramp-limited reachable set rather than a bound
that could turn out to be infeasible later in the day.

The shield uses the FORECAST baseline, because that is what a controller knows
when it sets a setpoint. Settlement then uses the REALISED demand, with the
physical floor at zero. A plan that assumed demand which did not arrive gets
clipped, and loses the money -- exactly as it would on site.
"""
from __future__ import annotations

import numpy as np

H = 96
DT_H = 0.25

U_LO, U_HI = -200.0, 200.0
EV1_LO, EV1_HI = 0.0, 1200.0
EV2_LO, EV2_HI = 0.0, 1800.0
RAMP_MAX = 80.0

OBS_DIM = 22
ACT_DIM = 2


def reachable_sum(u_now, steps, lo, hi):
    """Total that the remaining `steps` can still contribute, ramp-limited.

    Returns (most_negative, most_positive). Starting from `u_now`, the fastest
    the setpoint can move is RAMP_MAX per step, so the extreme cumulative
    contribution is the sum of the ramp-limited envelope, saturated at the box.
    Using the true envelope instead of steps*U_HI is what keeps the closure
    feasible: an optimistic bound lets the agent overspend early and then makes
    sum(u) = 0 unreachable, which would silently break the energy constraint.
    """
    if steps <= 0:
        return 0.0, 0.0
    k = np.arange(1, steps + 1)
    up = np.minimum(hi, u_now + RAMP_MAX * k).sum()
    dn = np.maximum(lo, u_now - RAMP_MAX * k).sum()
    return float(dn), float(up)


def admissible(u_prev, budget, base_t, steps_left, ev_hi):
    """The interval this step's setpoint may take, for one site."""
    lo = max(U_LO, u_prev - RAMP_MAX, -base_t)
    hi = min(U_HI, u_prev + RAMP_MAX, ev_hi - base_t)
    if lo > hi:                       # degenerate baseline; stay put if legal
        lo = hi = float(np.clip(0.0, U_LO, U_HI))

    # after acting, the rest of the day must still be able to return `budget-u`
    dn, up = reachable_sum(0.0, steps_left, U_LO, U_HI)
    lo = max(lo, budget - up)
    hi = min(hi, budget - dn)
    if lo > hi:
        lo = hi = float(np.clip(budget, lo, hi)) if steps_left == 0 else float(
            np.clip(0.0, min(lo, hi), max(lo, hi)))
    return lo, hi


class SiteEnv:
    """Vector-free, single-episode environment. No gym dependency."""

    def __init__(self, ev1, ev2, pv, f1, f2, price, rng=None,
                 resid1=None, resid2=None, price_dates=None, dates=None):
        self.ev1, self.ev2, self.pv = ev1, ev2, pv
        self.f1, self.f2 = f1, f2
        self.price = price
        # Forecast residuals, ALWAYS taken from the training days, so the
        # stochastic arm never sees the distribution of the days it is scored
        # on. Kept per time-of-day slot: the uncertainty at 03:00, when the
        # chargers are idle, is not the uncertainty at 18:00.
        self.resid1, self.resid2 = resid1, resid2
        self.price_dates, self.dates = price_dates, dates
        self.rng = rng or np.random.default_rng(0)
        self.n_days = len(ev1)
        self.n_price = len(price)
        self._day = None
        self._pday = None

    # ------------------------------------------------------------ episode --
    def reset(self, day=None, price_day=None):
        self._day = int(self.rng.integers(self.n_days)) if day is None else int(day)
        self._pday = (int(self.rng.integers(self.n_price))
                      if price_day is None else int(price_day))

        self.r1 = self.ev1[self._day]
        self.r2 = self.ev2[self._day]
        self.b1 = self.f1[self._day]
        self.b2 = self.f2[self._day]
        self.pvd = self.pv[self._day]
        self.p = self.price[self._pday]

        self.t = 0
        self.u_prev = np.zeros(2)
        self.budget = np.zeros(2)          # what still must be returned
        self.u_log = np.zeros((H, 2))
        self.pz = (self.p - self.p.mean()) / (self.p.std() + 1e-6)
        return self._obs()

    def _obs(self):
        t = self.t
        left = H - t
        fut = slice(t, H)
        seen = slice(0, max(t, 1))

        fc_now = self.b1[t] + self.b2[t]
        err_so_far = (float(np.sum(self.r1[seen] + self.r2[seen]))
                      - float(np.sum(self.b1[seen] + self.b2[seen]))) if t else 0.0

        return np.array([
            t / H,
            np.sin(2 * np.pi * t / H), np.cos(2 * np.pi * t / H),
            self.pz[t],
            float(np.mean(self.pz[fut])),
            float(np.max(self.pz[fut])), float(np.min(self.pz[fut])),
            float((self.pz[t] - np.mean(self.pz[fut])) / (np.std(self.pz[fut]) + 1e-6)),
            float(np.mean(self.p[fut]) / 100.0),
            self.p[t] / 100.0,
            self.pvd[t] / 500.0,
            float(np.mean(self.pvd[fut]) / 500.0),
            fc_now / 500.0,
            float(np.mean(self.b1[fut] + self.b2[fut]) / 500.0),
            err_so_far / (500.0 * max(t, 1)),          # realised-vs-forecast drift
            self.budget[0] / (U_HI * 10.0), self.budget[1] / (U_HI * 10.0),
            self.u_prev[0] / U_HI, self.u_prev[1] / U_HI,
            left / H,
            float(np.clip(self.budget.sum() / (left * U_HI + 1e-6), -1, 1)),
            1.0,
        ], dtype=np.float32)

    # --------------------------------------------------------------- step --
    def step(self, action):
        """action in [-1, 1]^2. Returns (obs, reward_eur, done, info)."""
        t = self.t
        steps_left = H - t - 1
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # The action IS the setpoint, scaled, then clipped into the admissible
        # interval -- not an interpolation between its ends. That distinction
        # decides whether the agent can express "do nothing": under an
        # interpolation, a = 0 lands on the MIDPOINT of [lo, hi], so a policy
        # emitting zeros acts arbitrarily, and late in the day, when closing
        # the energy budget squeezes the interval, that midpoint is extreme.
        # Mapping directly makes a = 0 mean u = 0 whenever u = 0 is feasible,
        # which is both the natural neutral policy and a far better-conditioned
        # action space to explore from.
        u = np.zeros(2)
        for i, (prev, bud, base, hi_cap) in enumerate((
                (self.u_prev[0], self.budget[0], self.b1[t], EV1_HI),
                (self.u_prev[1], self.budget[1], self.b2[t], EV2_HI))):
            lo, hi = admissible(prev, bud, base, steps_left, hi_cap)
            u[i] = float(np.clip(a[i] * U_HI, lo, hi))

        # settle this step against what actually arrived, with the physical floor
        ev_ctrl = (max(self.r1[t] + u[0], 0.0) + max(self.r2[t] + u[1], 0.0))
        ev_base = self.r1[t] + self.r2[t]
        reward = float(self.p[t] * (ev_base - ev_ctrl) * DT_H / 1000.0)

        self.u_log[t] = u
        self.budget = self.budget - u          # u spent now must be returned
        self.u_prev = u
        self.t += 1
        done = self.t >= H
        info = {}
        if done:
            info["sum_u"] = self.u_log.sum(axis=0)
            info["day"] = self._day
            info["price_day"] = self._pday
        return (self._obs() if not done else np.zeros(OBS_DIM, np.float32),
                reward, done, info)

    # ------------------------------------------------------------ helpers --
    def settle_sequence(self, u1, u2, day, price_day):
        """Euros banked by an arbitrary plan on a given day. Used by baselines."""
        r1, r2, p = self.ev1[day], self.ev2[day], self.price[price_day]
        ev = np.maximum(r1 + u1, 0.0) + np.maximum(r2 + u2, 0.0)
        return float(np.sum(p * ((r1 + r2) - ev) * DT_H / 1000.0))

    def base_cost(self, day, price_day):
        r1, r2, p = self.ev1[day], self.ev2[day], self.price[price_day]
        return float(np.sum(p * ((r1 + r2) - self.pv[day]) * DT_H / 1000.0))


def load(datadir="rl/data", split="train", eval_days=120, seed=0):
    """Chronological split: the evaluation days are the LAST ones. No leakage."""
    d = np.load("%s/site_days.npz" % datadir, allow_pickle=True)
    p = np.load("%s/prices.npz" % datadir, allow_pickle=True)
    n = len(d["ev1"])
    cut = n - eval_days
    tr = slice(0, cut)
    sl = tr if split == "train" else slice(cut, n)
    resid1 = d["ev1"][tr] - d["f1"][tr]
    resid2 = d["ev2"][tr] - d["f2"][tr]
    env = SiteEnv(d["ev1"][sl], d["ev2"][sl], d["pv"][sl],
                  d["f1"][sl], d["f2"][sl], p["price"],
                  rng=np.random.default_rng(seed),
                  resid1=resid1, resid2=resid2,
                  price_dates=p["dates"], dates=d["dates"][sl])
    return env, d["dates"][sl], p["dates"]


def dated_pairs(env, n_days=None):
    """Pair each demand day with the price day of the SAME DATE.

    Drawing a price day at random from four years is right for TRAINING -- it
    turns 487 days into hundreds of thousands of scenarios and forces the
    policy to answer the price shape rather than memorise a date. It is wrong
    for EVALUATION: it lands 2022 crisis prices, which touched 3000 EUR/MWh, on
    2025 demand, inflating every spread and every extreme with a combination
    that never occurred. Scoring uses the price the site actually faced.
    """
    idx = {str(dt): i for i, dt in enumerate(env.price_dates)}
    out = []
    for i, dt in enumerate(env.dates):
        j = idx.get(str(dt))
        if j is not None:
            out.append((i, j))
        if n_days is not None and len(out) >= n_days:
            break
    return out
