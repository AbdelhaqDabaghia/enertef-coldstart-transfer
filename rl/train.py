"""
train.py -- train the policy, and check it against the MPC while it trains.

Every `eval_every` steps the current policy is run deterministically over a
fixed set of held-out days paired with fixed price days, and compared to
`mpc_ledger` on exactly those pairs. The comparison is paired from the first
line of output, because a mean over different days tells you nothing here: the
day-to-day spread is 18 % of daily cost and the effect being chased is about
1 %.

The checkpoint kept is the one with the best HELD-OUT score, not the best
training score. Selecting on training reward is how the earlier CEM run
convinced itself it had learned something it had not.

    python -m rl.train --steps 600000

Everything is on CPU by default; this machine has no CUDA. 600k steps is
roughly 6250 episodes, which is about 17 passes over the training days.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from rl.site_env import load, OBS_DIM, ACT_DIM, H
from rl.sac import SAC
from rl.baselines import mpc_ledger


def fixed_pairs(env, n_days, n_prices, seed=12345):
    """Deterministic (demand day, price day) pairs for evaluation."""
    rng = np.random.default_rng(seed)
    days = np.arange(min(n_days, env.n_days))
    prices = rng.integers(0, env.n_price, size=len(days))
    return list(zip(days, prices))


def rollout_policy(agent, env, day, price_day):
    obs = env.reset(day=day, price_day=price_day)
    total = 0.0
    for _ in range(H):
        a = agent.act(obs, deterministic=True)
        obs, r, done, _ = env.step(a)
        total += r
        if done:
            break
    return total, env.u_log[:, 0].copy(), env.u_log[:, 1].copy()


def evaluate(agent, env, pairs, cache):
    """Paired policy-vs-MPC on the held-out pairs. Returns per-pair differences."""
    pol, mpc = [], []
    for day, pday in pairs:
        p, _, _ = rollout_policy(agent, env, day, pday)
        key = (int(day), int(pday))
        if key not in cache:
            u1, u2 = mpc_ledger(env, day, pday)
            cache[key] = env.settle_sequence(u1, u2, day, pday)
        pol.append(p)
        mpc.append(cache[key])
    pol, mpc = np.array(pol), np.array(mpc)
    return pol, mpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--start-steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    # Undiscounted: the objective is the day's TOTAL bill, every quarter-hour
    # counts the same, and the horizon is finite and fixed at 96 steps with the
    # step index in the observation. Discounting would quietly prefer acting
    # early, which the sum(u) = 0 budget then punishes at the close.
    ap.add_argument("--gamma", type=float, default=1.0)
    # Per-step rewards are around 0.15 EUR. At that scale SAC's entropy term
    # dominates the critic and the policy stays timid -- which is exactly the
    # plateau observed. Scaling the reward changes no optimum, only the
    # conditioning of the learning problem.
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-days", type=int, default=40)
    ap.add_argument("--holdout-days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="rl/runs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = "sac_s%d" % args.seed
    ckpt = os.path.join(args.outdir, tag + ".pt")
    logf = os.path.join(args.outdir, tag + "_log.jsonl")

    train_env, train_dates, price_dates = load(split="train",
                                               eval_days=args.holdout_days,
                                               seed=args.seed)
    val_env, val_dates, _ = load(split="eval", eval_days=args.holdout_days,
                                 seed=args.seed + 777)
    print("[train] %d training days (%s -> %s)"
          % (train_env.n_days, train_dates[0], train_dates[-1]), flush=True)
    print("[train] %d held-out days (%s -> %s) | %d price days"
          % (val_env.n_days, val_dates[0], val_dates[-1], val_env.n_price),
          flush=True)
    print("[train] scenario space %d x %d = %d pairs"
          % (train_env.n_days, train_env.n_price,
             train_env.n_days * train_env.n_price), flush=True)

    agent = SAC(OBS_DIM, ACT_DIM, hidden=args.hidden, lr=args.lr,
                gamma=args.gamma, device=args.device, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    pairs = fixed_pairs(val_env, args.eval_days, val_env.n_price)
    mpc_cache = {}

    obs = train_env.reset()
    ep_ret, ep_n, best = 0.0, 0, -np.inf
    t0 = time.time()
    returns = []

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            act = rng.uniform(-1, 1, size=ACT_DIM)
        else:
            act = agent.act(obs)

        obs2, rew, done, _ = train_env.step(act)
        # The end of the day IS terminal. The next day is an independent
        # episode with no carried state, so bootstrapping across the boundary
        # would value the close of one day with the start of an unrelated one.
        # Storing done=0 here -- as this did -- injects that noise into every
        # target, and it is also what makes gamma = 1 safe: a finite horizon
        # that genuinely terminates cannot diverge.
        agent.buf.add(obs, act, rew * args.reward_scale, obs2,
                      1.0 if done else 0.0)
        obs = obs2
        ep_ret += rew

        if done:
            returns.append(ep_ret)
            ep_ret, ep_n = 0.0, ep_n + 1
            obs = train_env.reset()

        if step > args.start_steps:
            for _ in range(args.updates_per_step):
                agent.update(args.batch)

        if step % args.eval_every == 0:
            pol, mpc = evaluate(agent, val_env, pairs, mpc_cache)
            d = pol - mpc
            rec = dict(step=step, episodes=ep_n,
                       train_ret=float(np.mean(returns[-50:])) if returns else 0.0,
                       policy=float(pol.mean()), mpc=float(mpc.mean()),
                       diff=float(d.mean()),
                       dz=float(d.mean() / (d.std(ddof=1) + 1e-9)),
                       wins=int((d > 0).sum()), n=len(d),
                       mins=round((time.time() - t0) / 60.0, 1))
            with open(logf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("[%7d] train %+7.2f | HELD-OUT policy %+7.2f  mpc %+7.2f  "
                  "diff %+7.3f  wins %2d/%d  (%.1f min)"
                  % (step, rec["train_ret"], rec["policy"], rec["mpc"],
                     rec["diff"], rec["wins"], rec["n"], rec["mins"]), flush=True)
            if rec["diff"] > best:
                best = rec["diff"]
                agent.save(ckpt)
                print("          ^ best held-out margin so far, checkpoint saved",
                      flush=True)

    print("\n[train] done in %.1f min. best held-out margin %+.3f EUR/day"
          % ((time.time() - t0) / 60.0, best))
    print("[train] checkpoint %s" % ckpt)
    print("[train] now run:  python -m rl.evaluate --ckpt %s" % ckpt)


if __name__ == "__main__":
    main()
