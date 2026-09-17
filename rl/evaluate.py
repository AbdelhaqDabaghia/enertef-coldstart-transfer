"""
evaluate.py -- the final answer: which controller banks more, on held-out days.

Five arms on the SAME (demand day, price day) pairs, settled by the same
ledger against the demand that actually arrived:

    mpc_deployed   what the site runs today
    mpc_ledger     same program, same information, correct objective
    mpc_rh         receding horizon -- the honest closed-loop baseline
    rl             the trained policy
    oracle         perfect demand knowledge; the ceiling

Everything is paired and reported with Wilcoxon and paired d_z, because the
day-to-day spread here is ~18 % of daily cost while the differences in play are
around 1 %: an unpaired mean would be noise.

    python -m rl.evaluate --ckpt rl/runs/sac_s0.pt

Writes Data/results/e24_rl_project.csv.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from rl.site_env import load, dated_pairs, OBS_DIM, ACT_DIM, H
from rl.sac import SAC
from rl.baselines import ARMS


def paired_stats(d):
    d = np.asarray(d, float)
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:
        p = float("nan")
    return dz, p, int((d > 0).sum()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rl/runs/sac_s0.pt")
    ap.add_argument("--holdout-days", type=int, default=120)
    ap.add_argument("--eval-days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default="Data/results/e24_rl_project.csv")
    args = ap.parse_args()

    env, dates, price_dates = load(split="eval", eval_days=args.holdout_days,
                                   seed=args.seed)
    pairs = dated_pairs(env, args.eval_days)
    n = len(pairs)
    print("[eval] %d held-out days: %s -> %s" % (n, dates[0], dates[n - 1]),
          flush=True)

    agent = SAC(OBS_DIM, ACT_DIM, hidden=args.hidden, device="cpu")
    have_rl = os.path.exists(args.ckpt)
    if have_rl:
        agent.load(args.ckpt)
        print("[eval] loaded %s" % args.ckpt, flush=True)
    else:
        print("[eval] WARNING: %s not found -- reporting baselines only"
              % args.ckpt, flush=True)

    rows = []
    for k, (day, pday) in enumerate(pairs):
        row = dict(day=int(day), date=str(dates[day]),
                   price_date=str(price_dates[pday]),
                   base=round(env.base_cost(day, pday), 4),
                   price_spread=round(float(env.price[pday].max()
                                            - env.price[pday].min()), 2))
        for name, fn in ARMS.items():
            u1, u2 = fn(env, day, pday)
            row[name] = round(env.settle_sequence(u1, u2, day, pday), 4)
        if have_rl:
            obs = env.reset(day=day, price_day=pday)
            tot = 0.0
            for _ in range(H):
                obs, r, done, _ = env.step(agent.act(obs, deterministic=True))
                tot += r
                if done:
                    break
            row["rl"] = round(tot, 4)
        rows.append(row)
        if (k + 1) % 20 == 0:
            print("  ... %d/%d days" % (k + 1, n), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    arms = [a for a in ("mpc_deployed", "mpc_ledger", "mpc_rh", "mpc_saa",
                        "rl", "oracle") if a in df.columns]
    print("\n=== EUR banked per day vs an unmanaged site, %d held-out days ===" % n)
    for a in arms:
        v = df[a].to_numpy()
        print("  %-13s mean %+8.3f   SD %7.3f   min %+8.3f   max %+8.3f"
              % (a, v.mean(), v.std(ddof=1), v.min(), v.max()))
    print("  %-13s mean %8.3f  (cost of doing nothing)"
          % ("base", df["base"].mean()))

    print("\n=== paired differences ===")
    comparisons = [("mpc_ledger", "mpc_deployed"), ("mpc_rh", "mpc_ledger"),
                   ("mpc_saa", "mpc_ledger")]
    if "rl" in df.columns:
        comparisons += [("rl", "mpc_deployed"), ("rl", "mpc_ledger"),
                        ("rl", "mpc_saa")]
    comparisons += [("oracle", "mpc_deployed")]
    for a, b in comparisons:
        d = (df[a] - df[b]).to_numpy()
        dz, p, wins, nn = paired_stats(d)
        print("  %-26s mean %+8.3f EUR/day  dz %+6.3f  p %.4f  wins %3d/%d  "
              "(%+.2f %% of base)"
              % ("%s - %s" % (a, b), d.mean(), dz, p, wins, nn,
                 100 * d.mean() / df["base"].mean()))

    ceiling = df["oracle"].mean() - df["mpc_deployed"].mean()
    print("\n=== headroom above the deployed controller ===")
    print("  oracle - mpc_deployed = %+.3f EUR/day  (%.2f %% of base)"
          % (ceiling, 100 * ceiling / df["base"].mean()))
    if ceiling > 1e-9:
        for a in arms:
            if a in ("mpc_deployed", "oracle"):
                continue
            got = df[a].mean() - df["mpc_deployed"].mean()
            print("    %-13s recovers %6.1f %% of it" % (a, 100 * got / ceiling))
    print("\n[eval] wrote %s" % args.out)


if __name__ == "__main__":
    main()
