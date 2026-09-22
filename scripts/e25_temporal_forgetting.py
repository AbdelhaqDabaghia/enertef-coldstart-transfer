"""
e25_temporal_forgetting.py -- does the DEPLOYED nightly loop actually forget?

WHY THIS EXPERIMENT EXISTS. Every continual-learning result in this project so
far (e13, e15, e16) is cross-domain TRANSFER: a model pre-trained on the
Luxembourg site, adapted to the UK Electric Nation trial, with "retention"
measured on the Luxembourg holdout. That answers a real question, but it is not
the question the deployed system poses. The nightly job adapts on NEW
LUXEMBOURG DATA and is meant to track seasonal drift, behaviour change and a
growing charger fleet at the SAME site. Temporal forgetting at one site and
spatial forgetting across two sites are different phenomena, and the second has
never been measured here.

THE PROTOCOL. The historical record is cut into consecutive windows. The model
walks forward through them exactly as the nightly job does -- each cycle
warm-starts from the previous cycle's weights, never from production -- and
after every cycle it is scored on the holdout of EVERY window seen so far. That
gives the full backward-transfer matrix: row = cycle, column = window, and
everything above the diagonal is what adaptation did to knowledge it already
had.

  forgetting(k) = mean over j < k of [ nRMSE(model_k, window_j)
                                     - nRMSE(model_j, window_j) ]

Positive means the model got worse on regimes it had already learned. That is
the definition the CL literature uses, and it is the number the deployed
system's premise depends on.

THE CONTROLLER IS NOT TOUCHED. Each cycle's forecast is also pushed through
solve_ecc_mpc_two_sites, transcribed verbatim from ECC_MCP_v0.5.ipynb cell 48 --
the same objective, weights, bounds, ramp and sum(u)=0 constraints that run on
site, with prices in EUR/MWh. Not the ledger or stochastic variants from the
rl/ project: those are a different study, and mixing them would make this
result incomparable to what the site actually does.

Conditions, all through coldstart_transfer.trainer.run_condition so the code
path is byte-for-byte the one behind every other number in the paper:

    B3  warm-start fine-tuning, no protection   -- the thing said to forget
    B1  output-sensitivity (MAS-style) proximal regulariser
    B5  bounded rehearsal with reservoir sampling

Writes Data/results/e25_temporal_forgetting.csv (one row per cycle x eval
window x condition x seed) and e25_temporal_summary.csv.

    PYTHONPATH=. python scripts/e25_temporal_forgetting.py --cycles 3 --seeds 1
    PYTHONPATH=. python scripts/e25_temporal_forgetting.py   # full campaign
"""
from __future__ import annotations

import argparse
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

from coldstart_transfer.windowing import make_windows, LOOKBACK, inverse_target
from coldstart_transfer.trainer import run_condition, nrmse

FEATURES = os.environ.get("FEATURES", "Data/lux_source_features_causal.csv")
SCALERS = os.environ.get("SCALERS", "Data/models/ev_scalers_causal_full_warm_e20_s1.joblib")
PROD = os.environ.get("PROD", "Data/models/ev_cnn_lstm_causal_full_warm_e20_s1.keras")
PRICES = os.environ.get("PRICES", "Data/entsoe_dayahead_DE_LU_4y.csv")
OUT = os.environ.get("OUT", "Data/results/e25_temporal_forgetting.csv")

WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "30"))
HOLDOUT_DAYS = int(os.environ.get("HOLDOUT_DAYS", "7"))

CONDITIONS = {"B3": (0.0, 0.0), "B1": (1.0, 0.0), "B5": (0.0, 1.0)}
PAST_DAYS = int(os.environ.get("PAST_DAYS", "60"))   # bounded replay, as deployed

# --- the deployed controller, verbatim from ECC_MCP_v0.5.ipynb cell 48 -------
U_LO, U_HI = -200.0, 200.0
EV1_LO, EV1_HI = 0.0, 1200.0
EV2_LO, EV2_HI = 0.0, 1800.0
RAMP_MAX = 80.0
W_IMPORT, W_EXPORT, W_U, W_RAMP = 8.0, 2.0, 1e-4, 0.01
CAP1, CAP2 = 4 * 300.0, 4 * 400.0 + 8 * 22.0
SHARE1 = CAP1 / (CAP1 + CAP2)
SHARE2 = 1.0 - SHARE1
DT_H, H = 0.25, 96


def solve_two_sites(pv, ev1b, ev2b, price):
    """Verbatim transcription. Do not substitute a different objective here."""
    import cvxpy as cp
    n = len(pv)
    u1, u2 = cp.Variable(n), cp.Variable(n)
    ev1c, ev2c = ev1b + u1, ev2b + u2
    pgrid = ev1c + ev2c - pv
    imp, exp = cp.Variable(n), cp.Variable(n)
    cons = [imp >= 0, exp >= 0, pgrid == imp - exp,
            ev1c >= EV1_LO, ev1c <= EV1_HI, ev2c >= EV2_LO, ev2c <= EV2_HI,
            u1 >= U_LO, u1 <= U_HI, u2 >= U_LO, u2 <= U_HI,
            cp.abs(u1[1:] - u1[:-1]) <= RAMP_MAX,
            cp.abs(u2[1:] - u2[:-1]) <= RAMP_MAX,
            cp.sum(u1) == 0, cp.sum(u2) == 0]
    obj = (W_IMPORT * cp.sum(cp.multiply(price, imp * DT_H / 1000.0))
           - W_EXPORT * cp.sum(cp.multiply(price, exp * DT_H / 1000.0))
           + W_U * (cp.sum_squares(u1) + cp.sum_squares(u2))
           + W_RAMP * (cp.sum_squares(u1[1:] - u1[:-1])
                       + cp.sum_squares(u2[1:] - u2[:-1])))
    try:
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.OSQP, verbose=False)
    except Exception:
        return np.zeros(n), np.zeros(n)
    if u1.value is None:
        return np.zeros(n), np.zeros(n)
    return np.asarray(u1.value).ravel(), np.asarray(u2.value).ravel()


def bill(ev_total, pv, price):
    pg = ev_total - pv
    return float(np.sum(np.clip(pg, 0, None) * DT_H / 1000.0 * price)
                 - np.sum(np.clip(-pg, 0, None) * DT_H / 1000.0 * price))


def settle(model, holdout, scalers, pv, price):
    """Plan on the model's forecast, settle against the demand that arrived."""
    Xs, Xn, y, _ = holdout
    n = (len(y) // H) * H
    if n == 0:
        return float("nan")
    fc = np.clip(inverse_target(
        np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None),
        scalers), 0, None)[:n]
    real = inverse_target(y, scalers)[:n]
    tot = 0.0
    for d in range(n // H):
        s = slice(d * H, (d + 1) * H)
        p, pvd = price[s], pv[s]
        u1, u2 = solve_two_sites(pvd, SHARE1 * fc[s], SHARE2 * fc[s], p)
        ev = (np.clip(SHARE1 * real[s] + u1, 0, None)
              + np.clip(SHARE2 * real[s] + u2, 0, None))
        tot += bill(real[s], pvd, p) - bill(ev, pvd, p)
    return tot / max(n // H, 1)


def build_windows(scalers, n_windows):
    """Consecutive (train, holdout) windows walking forward through history."""
    sdf = pd.read_csv(FEATURES)
    Xs, Xn, y, ts = make_windows(sdf, scalers)
    per = WINDOW_DAYS * 96
    hold = HOLDOUT_DAYS * 96
    total = len(y) // per
    if n_windows:
        total = min(total, n_windows)
    wins = []
    for k in range(total):
        a, b = k * per, (k + 1) * per
        cut = b - hold
        wins.append(dict(
            k=k,
            train=(Xs[a:cut], Xn[a:cut], y[a:cut], ts[a:cut]),
            hold=(Xs[cut:b], Xn[cut:b], y[cut:b], ts[cut:b]),
            t0=str(pd.to_datetime(ts[a], utc=True).date()),
            t1=str(pd.to_datetime(ts[b - 1], utc=True).date()),
        ))
    return wins


def load_exogenous(wins):
    """PV and prices aligned to each window's holdout, by DATE not position."""
    site = pd.read_csv("Data/ECC_master_PV_EMOB1_EMOB2_15min.csv",
                       parse_dates=["Started at"])
    site["t"] = pd.to_datetime(site["Started at"], utc=True)
    pv = site.set_index("t")["PV_TotalProduction_kW"].astype(float)

    pr = None
    if os.path.exists(PRICES):
        p = pd.read_csv(PRICES)
        p["t"] = pd.to_datetime(p["timestamp"], utc=True)
        pr = p.set_index("t")["price"].astype(float)

    for w in wins:
        ts = pd.to_datetime(w["hold"][3], utc=True)
        w["pv"] = pv.reindex(ts).ffill().fillna(0.0).to_numpy()
        w["price"] = (pr.reindex(ts).ffill().bfill().to_numpy()
                      if pr is not None else np.full(len(ts), 100.0))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=0, help="0 = all available")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-mpc", action="store_true")
    args = ap.parse_args()

    scalers = joblib.load(SCALERS)
    wins = load_exogenous(build_windows(scalers, args.cycles))
    print("[e25] %d windows of %d days (holdout %d): %s -> %s"
          % (len(wins), WINDOW_DAYS, HOLDOUT_DAYS, wins[0]["t0"], wins[-1]["t1"]),
          flush=True)

    rows = []
    tmp = tempfile.mkdtemp(prefix="e25_")
    for cond, (lam, beta) in CONDITIONS.items():
        for seed in range(args.seeds):
            # Each cycle warm-starts from the PREVIOUS cycle, never from
            # production. That chaining is what makes this a task stream rather
            # than a set of independent adaptations.
            warm = PROD
            diag = {}                       # nRMSE on window j when j was learned
            for w in wins:
                # The "source" a temporal cycle must protect is the PAST, not
                # the window being learned. Passing the current window here --
                # as the first draft did -- makes replay re-inject the data it
                # is already training on and makes the Fisher penalty guard the
                # current task, so all three conditions collapse onto plain
                # fine-tuning. The smoke run showed exactly that: B3 and B1
                # identical to four decimals.
                #
                # Bounded to the most recent PAST_DAYS of prior windows, which
                # is what the deployed replay buffer holds -- an unbounded
                # buffer would be a different method from the one on site.
                past = wins[:w["k"]]
                if past:
                    keep = (PAST_DAYS * 96) // max(len(past), 1)
                    src = tuple(np.concatenate([p_["train"][i][-keep:]
                                                for p_ in past]) for i in range(4))
                    src_hold = past[0]["hold"]        # the OLDEST regime
                else:
                    src, src_hold = w["train"], w["hold"]
                r = run_condition(
                    "B3" if cond == "B3" else ("B1" if cond == "B1" else "B5"),
                    w["train"], w["hold"], scalers, warm,
                    n_days=WINDOW_DAYS, lambda_ewc=lam, beta_replay=beta,
                    seed=seed, epochs=args.epochs, batch_size=64, lr=1e-4,
                    source_train=src, source_holdout=src_hold)
                model = r.pop("model")

                # score on every window seen so far -- the backward matrix
                for prev in wins[:w["k"] + 1]:
                    e, _, _ = nrmse(model, prev["hold"], scalers)
                    if prev["k"] == w["k"]:
                        diag[prev["k"]] = e
                    rows.append(dict(
                        condition=cond, seed=seed, cycle=w["k"],
                        eval_window=prev["k"], window_start=prev["t0"],
                        nrmse=round(e, 6),
                        learned_nrmse=round(diag.get(prev["k"], np.nan), 6),
                        forgetting=round(e - diag.get(prev["k"], np.nan), 6),
                        savings_eur_day=round(
                            settle(model, prev["hold"], scalers,
                                   prev["pv"], prev["price"]), 4)
                        if (not args.no_mpc and prev["k"] == w["k"]) else np.nan,
                    ))

                warm = os.path.join(tmp, "%s_s%d_c%d.keras" % (cond, seed, w["k"]))
                model.save(warm)
                print("  [%s s=%d cycle %2d %s] now %.4f | forgetting %+0.4f"
                      % (cond, seed, w["k"], w["t0"], diag[w["k"]],
                         np.nanmean([r_["forgetting"] for r_ in rows
                                     if r_["condition"] == cond
                                     and r_["seed"] == seed
                                     and r_["cycle"] == w["k"]
                                     and r_["eval_window"] < w["k"]] or [np.nan])),
                      flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    past = df[df.eval_window < df.cycle]
    print("\n=== FORGETTING: error on regimes already learned, vs when learned ===")
    print("    positive = the model got worse on what it already knew")
    for c in CONDITIONS:
        v = past[past.condition == c]["forgetting"]
        if len(v):
            print("  %-4s mean %+0.4f nRMSE   SD %.4f   n=%d"
                  % (c, v.mean(), v.std(ddof=1), len(v)))
    print("\n=== ADAPTATION: error on the window just learned ===")
    cur = df[df.eval_window == df.cycle]
    for c in CONDITIONS:
        v = cur[cur.condition == c]["nrmse"]
        if len(v):
            print("  %-4s mean %.4f nRMSE   SD %.4f" % (c, v.mean(), v.std(ddof=1)))
    if not args.no_mpc:
        print("\n=== DECISION: EUR/day banked, deployed MPC on each forecast ===")
        for c in CONDITIONS:
            v = cur[cur.condition == c]["savings_eur_day"].dropna()
            if len(v):
                print("  %-4s mean %+8.3f EUR/day   SD %.3f" % (c, v.mean(), v.std(ddof=1)))
    print("\n[e25] wrote %s" % OUT)


if __name__ == "__main__":
    main()
