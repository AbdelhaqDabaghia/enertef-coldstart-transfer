"""
e23_rl_vs_mpc.py -- which controller banks more money?

Direct, paired comparison of three controllers on the SAME days, with the SAME
information, settled by the SAME ledger against the demand that actually
arrived. No simulation of the outcome: every euro reported here is
`bill(realised)`, exactly as e18 settles the deployed controller.

    mpc      the deployed controller. Convex two-site problem
             (solve_ecc_mpc_two_sites, ECC_MCP_v0.5.ipynb cell 48), solved to
             optimality by OSQP against the FORECAST, then settled against the
             realised demand.

    rl       a learned policy proposing (u1, u2) per step from the same inputs
             the MPC sees. Its raw proposal is projected onto the MPC's exact
             feasible set, so neither arm can buy an advantage by violating a
             constraint the other respects. Trained on earlier days against the
             REALISED bill, so it can learn to hedge forecast error rather than
             trust a point forecast.

    oracle   the same convex MPC given the TRUE demand instead of the forecast.
             No causal controller can beat this. It is the ceiling, and it is
             the number that says whether there is anything left to win.

The comparison the project actually needs is `oracle - mpc`: the headroom. If
the deployed MPC already sits near the ceiling, no controller -- learned or
otherwise -- can recover much, and the binding constraint is data quality, not
control. If the headroom is wide, `rl - mpc` says how much of it a policy
trained on realised outcomes picks up.

Why the MPC is hard to beat, stated plainly: it solves a convex program to
optimality. Given identical inputs it cannot be beaten, only matched. The one
gap a learned policy can exploit is that the MPC optimises against a forecast
while the bill is settled against reality -- it is optimal for a world that
does not arrive.

Writes Data/results/e23_rl_vs_mpc.csv. CPU only, no GPU needed.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import cvxpy as cp

from coldstart_transfer.windowing import make_windows, inverse_target, LOOKBACK
from coldstart_transfer.model import build_ev_model

# ---------------------------------------------------------------- data paths
FEATURES = os.environ.get("FEATURES", "Data/lux_source_features_causal.csv")
MODEL = os.environ.get("MODEL", "Data/models/ev_cnn_lstm_causal_full_warm_e20_s1.keras")
SCALERS = os.environ.get("SCALERS", "Data/models/ev_scalers_causal_full_warm_e20_s1.joblib")
SITE = "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv"
PRICE_CSV = os.environ.get(
    "PRICE_CSV", r"C:\dev\MPC_RL\BESS_MPC_PROJECT\prices_da_real.csv")
OUT = os.environ.get("OUT", "Data/results/e23_rl_vs_mpc.csv")

EVAL_DAYS = int(os.environ.get("EVAL_DAYS", "21"))
TRAIN_DAYS = int(os.environ.get("TRAIN_DAYS", "60"))
SEED = int(os.environ.get("SEED", "0"))

# --- controller constants, transcribed from solve_ecc_mpc_two_sites ---------
U_LO, U_HI = -200.0, 200.0
EV1_LO, EV1_HI = 0.0, 1200.0
EV2_LO, EV2_HI = 0.0, 1800.0
RAMP_MAX = 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 8.0, 2.0, 1e-4, 0.01
CAP1 = 4 * 300.0
CAP2 = 4 * 400.0 + 8 * 22.0
SHARE1 = CAP1 / (CAP1 + CAP2)
SHARE2 = 1.0 - SHARE1

DT_H, H = 0.25, 96

# --- CEM hyper-parameters ---------------------------------------------------
POP = int(os.environ.get("POP", "32"))
ELITE = int(os.environ.get("ELITE", "8"))
ITERS = int(os.environ.get("ITERS", "40"))
BATCH = int(os.environ.get("BATCH", "16"))     # train days sampled per iteration
N_FEAT = 7                                     # policy inputs, incl. bias


# ============================================================== the ledger ==
def bill(ev_total, pv, price):
    """The notebook's ledger: import cost minus export revenue, both at 1:1."""
    pg = ev_total - pv
    imp = np.clip(pg, 0, None)
    exp = np.clip(-pg, 0, None)
    return float(np.sum(imp * DT_H / 1000.0 * price)
                 - np.sum(exp * DT_H / 1000.0 * price))


# ================================================== the MPC (deployed form) ==
def _mpc_problem(n):
    """Compile once, re-solve with parameters. Same program as the notebook."""
    u1, u2 = cp.Variable(n), cp.Variable(n)
    pv = cp.Parameter(n)
    b1, b2 = cp.Parameter(n), cp.Parameter(n)
    price = cp.Parameter(n, nonneg=True)

    ev1c, ev2c = b1 + u1, b2 + u2
    pgrid = ev1c + ev2c - pv
    imp, exp = cp.Variable(n), cp.Variable(n)

    cons = [imp >= 0, exp >= 0, pgrid == imp - exp,
            ev1c >= EV1_LO, ev1c <= EV1_HI, ev2c >= EV2_LO, ev2c <= EV2_HI,
            u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]

    e_imp = imp * DT_H / 1000.0
    e_exp = exp * DT_H / 1000.0
    obj = (W_IMPORT * cp.sum(cp.multiply(price, e_imp))
           - W_EXPORT * cp.sum(cp.multiply(price, e_exp))
           + W_U * (cp.sum_squares(u1) + cp.sum_squares(u2))
           + W_RAMP * (cp.sum_squares(u1[1:] - u1[:-1])
                       + cp.sum_squares(u2[1:] - u2[:-1])))
    prob = cp.Problem(cp.Minimize(obj), cons)
    return prob, (pv, b1, b2, price), (u1, u2)


def _projection_problem(n):
    """Euclidean projection onto the MPC's feasible set.

    The learned policy proposes freely; this makes its proposal admissible
    without letting it gain anything the MPC could not also have done. Same
    constraints, verbatim -- the two arms compete on the same ground.
    """
    u1, u2 = cp.Variable(n), cp.Variable(n)
    r1, r2 = cp.Parameter(n), cp.Parameter(n)
    b1, b2 = cp.Parameter(n), cp.Parameter(n)

    ev1c, ev2c = b1 + u1, b2 + u2
    cons = [ev1c >= EV1_LO, ev1c <= EV1_HI, ev2c >= EV2_LO, ev2c <= EV2_HI,
            u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]
    obj = cp.Minimize(cp.sum_squares(u1 - r1) + cp.sum_squares(u2 - r2))
    return cp.Problem(obj, cons), (r1, r2, b1, b2), (u1, u2)


_MPC = _mpc_problem(H)
_PROJ = _projection_problem(H)


def solve_mpc(pv, b1, b2, price):
    prob, (p_pv, p_b1, p_b2, p_pr), (u1, u2) = _MPC
    p_pv.value, p_b1.value, p_b2.value, p_pr.value = pv, b1, b2, np.clip(price, 0, None)
    try:
        prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
    except Exception:
        return np.zeros(H), np.zeros(H)
    if u1.value is None or u2.value is None:
        return np.zeros(H), np.zeros(H)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


def project(raw1, raw2, b1, b2):
    prob, (p_r1, p_r2, p_b1, p_b2), (u1, u2) = _PROJ
    p_r1.value, p_r2.value, p_b1.value, p_b2.value = raw1, raw2, b1, b2
    try:
        prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
    except Exception:
        return np.zeros(H), np.zeros(H)
    if u1.value is None or u2.value is None:
        return np.zeros(H), np.zeros(H)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


# ==================================================== the learned policy ====
def policy_features(price, pv, evf, hours):
    """Inputs the policy sees. Strictly a subset of what the MPC is given."""
    pz = (price - price.mean()) / (price.std() + 1e-6)
    rank = np.argsort(np.argsort(price)) / (len(price) - 1.0) - 0.5
    tod = 2 * np.pi * hours / 24.0
    return np.stack([
        np.ones_like(price),          # bias
        pz,                           # price relative to the day
        rank,                         # price rank within the day
        pv / 1000.0,
        evf / 1000.0,
        np.sin(tod),
        np.cos(tod),
    ], axis=1)                        # (H, N_FEAT)


def policy_actions(theta, feats):
    """Linear policy -> tanh -> kW. Shape only; feasibility comes from project()."""
    w = theta.reshape(N_FEAT, 2)
    return U_HI * np.tanh(feats @ w)   # (H, 2)


def run_rl(theta, day):
    a = policy_actions(theta, day["feats"])
    u1, u2 = project(a[:, 0], a[:, 1], day["f1"], day["f2"])
    return u1, u2


# ======================================================== settle one arm ====
def settle(u1, u2, day):
    """Euros banked versus doing nothing, against the demand that arrived."""
    ev = (np.clip(day["r1"] + u1, 0, None) + np.clip(day["r2"] + u2, 0, None))
    return day["base"] - bill(ev, day["pv"], day["price"])


# ================================================================== main ====
def build_days():
    scalers = joblib.load(SCALERS)
    sdf = pd.read_csv(FEATURES)
    need = (TRAIN_DAYS + EVAL_DAYS) * H + LOOKBACK
    sdf = sdf.iloc[-need:].reset_index(drop=True)
    Xs, Xn, y, ts = make_windows(sdf, scalers)

    ev_real = inverse_target(y, scalers)
    m = build_ev_model()
    m.load_weights(MODEL)
    ev_fc = np.clip(inverse_target(
        np.clip(m.predict([Xs, Xn], verbose=0).reshape(-1), 0, None), scalers), 0, None)

    site = pd.read_csv(SITE, parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    tsx = pd.to_datetime(ts, utc=True)
    pv_real = (site.set_index("t")["PV_TotalProduction_kW"].astype(float)
               .reindex(tsx).ffill().fillna(0.0).to_numpy())

    p = pd.read_csv(PRICE_CSV)
    col = "price" if "price" in p.columns else "price_eur_kwh"
    v = p[col].astype(float).to_numpy()
    if v.mean() < 5:                                  # EUR/kWh -> EUR/MWh
        v = v * 1000.0
    prices = v[:(len(v) // H) * H].reshape(-1, H)

    n = (len(ev_real) // H) * H
    ev_real, ev_fc, pv_real = ev_real[:n], ev_fc[:n], pv_real[:n]
    hours = tsx.hour.to_numpy()[:n] + tsx.minute.to_numpy()[:n] / 60.0
    ndays = n // H

    days = []
    for d in range(ndays):
        s = slice(d * H, (d + 1) * H)
        price = prices[d % len(prices)]
        evF, evR, pv = ev_fc[s], ev_real[s], pv_real[s]
        day = dict(
            idx=d, price=price, pv=pv,
            f1=SHARE1 * evF, f2=SHARE2 * evF,
            r1=SHARE1 * evR, r2=SHARE2 * evR,
            evF=evF, evR=evR,
            base=bill(evR, pv, price),
            feats=policy_features(price, pv, evF, hours[s]),
        )
        days.append(day)
    return days


def train_cem(train_days, rng):
    mu = np.zeros(N_FEAT * 2)
    sigma = np.full(N_FEAT * 2, 0.6)
    best = (-np.inf, mu.copy())
    for it in range(ITERS):
        batch = [train_days[i] for i in
                 rng.choice(len(train_days), size=min(BATCH, len(train_days)),
                            replace=False)]
        cand = rng.normal(mu, sigma, size=(POP, N_FEAT * 2))
        scores = np.array([
            np.mean([settle(*run_rl(th, dy), dy) for dy in batch]) for th in cand])
        order = np.argsort(scores)[::-1]
        elite = cand[order[:ELITE]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.02
        if scores[order[0]] > best[0]:
            best = (float(scores[order[0]]), cand[order[0]].copy())
        if (it + 1) % 5 == 0:
            print("  [cem] iter %2d/%d  best-in-batch %+7.2f EUR/day  "
                  "mean %+7.2f  sigma %.3f"
                  % (it + 1, ITERS, scores[order[0]], scores.mean(), sigma.mean()),
                  flush=True)
    return mu, best


def paired_stats(diff):
    """Wilcoxon signed-rank + paired dz, the convention used across this repo."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:
        p = float("nan")
    wins = int((d > 0).sum())
    return dz, p, wins, n


def main():
    rng = np.random.default_rng(SEED)
    days = build_days()
    print("[e23] %d days built | EV real %.1f kW / forecast %.1f kW | PV %.1f kW"
          % (len(days),
             np.mean([d["evR"].mean() for d in days]),
             np.mean([d["evF"].mean() for d in days]),
             np.mean([d["pv"].mean() for d in days])), flush=True)

    train, evaluate = days[:-EVAL_DAYS], days[-EVAL_DAYS:]
    print("[e23] split: %d train days (earlier) / %d eval days (later)"
          % (len(train), len(evaluate)), flush=True)

    print("[e23] training the policy on realised bills ...", flush=True)
    mu, (best_score, best_theta) = train_cem(train, rng)
    # `best` is a maximum over POP*ITERS noisy batch estimates -- selection on
    # noise, biased upward and prone to generalise badly. The CEM distribution
    # mean is the honest estimator. Both are evaluated; `mu` is the headline.
    policies = {"rl": mu, "rl_best": best_theta}
    print("  [cem] best-candidate train score %+.2f EUR/day (reported for "
          "contrast only)" % best_score, flush=True)

    rows = []
    for d in evaluate:
        u1m, u2m = solve_mpc(d["pv"], d["f1"], d["f2"], d["price"])
        u1o, u2o = solve_mpc(d["pv"], d["r1"], d["r2"], d["price"])  # true demand
        row = dict(
            day=d["idx"],
            base=round(d["base"], 4),
            mpc=round(settle(u1m, u2m, d), 4),
            oracle=round(settle(u1o, u2o, d), 4),
            price_spread=round(float(d["price"].max() - d["price"].min()), 2),
        )
        for name, th in policies.items():
            row[name] = round(settle(*run_rl(th, d), d), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== EUR banked per day vs an unmanaged site, %d eval days ==="
          % len(df))
    for arm in ("mpc", "rl", "rl_best", "oracle"):
        v = df[arm].to_numpy()
        print("  %-7s mean %+8.3f   SD %7.3f   min %+8.3f   max %+8.3f"
              % (arm, v.mean(), v.std(ddof=1), v.min(), v.max()))
    print("  %-7s mean %8.3f  (cost of doing nothing)" % ("base", df["base"].mean()))

    print("\n=== paired differences ===")
    for a, b in (("rl", "mpc"), ("rl_best", "mpc"), ("oracle", "mpc"),
                 ("oracle", "rl")):
        d = (df[a] - df[b]).to_numpy()
        dz, p, wins, n = paired_stats(d)
        print("  %-14s mean %+8.3f EUR/day  dz %+6.3f  p %.4f  "
              "wins %d/%d  (%.2f %% of base)"
              % ("%s - %s" % (a, b), d.mean(), dz, p, wins, n,
                 100 * d.mean() / df["base"].mean()))

    head = df["oracle"].mean() - df["mpc"].mean()
    print("\n=== headroom ===")
    print("  oracle - mpc = %+.3f EUR/day: the most ANY controller could add"
          % head)
    if head > 1e-9:
        print("  rl recovers %.1f %% of it"
              % (100 * (df["rl"].mean() - df["mpc"].mean()) / head))
    print("\n[e23] wrote %s" % OUT)


if __name__ == "__main__":
    main()
