# RL controller project — does a learned policy bank more than the MPC?

A complete, self-contained experiment: build the dataset, train a Soft
Actor-Critic policy on the site's own history, and compare it against four
controllers on held-out days, settled by the ledger the site actually pays.

Everything here reports **money banked against an unmanaged site, on the demand
that actually arrived**. Nothing is a simulated outcome.

---

## Why a learned policy has a real opening here

The deployed MPC is a convex program solved to optimality. Given identical
inputs it cannot be beaten, only matched. So the question is what inputs and
what objective it actually has, and there the picture changes.

**1. It does not optimise the bill.** `bill()` is import cost minus export
revenue at the same price, and import minus export is just the grid position,
so the settled bill is `sum(price * (ev - pv) * dt)` — linear in the setpoints,
and savings are `-sum(price * u)`. The deployed controller instead minimises
`8*import - 2*export` plus quadratic regularisers. It is optimal for its own
objective, not for the invoice. `mpc_ledger` measures that gap.

**2. It is open loop.** One solve per day against a point forecast, no
recourse. `mpc_rh` re-solves as the day unfolds; without that baseline, a
closed-loop policy beating an open-loop plan would prove nothing.

**3. It trusts the forecast through the physical floor.** This is the big one,
and it is measured, not assumed. EV demand at this site is spiky and mostly
zero: on a typical held-out day, **52 of 96 steps have realised EMOB1 below
1 kW while the forecast says above 5 kW**. The MPC plans reductions against
demand that never arrives. Those reductions are truncated — you cannot charge
less than nothing — while the compensating increases apply in full. On the day
diagnosed in `rl/` development this turned a **+16.02 EUR planned saving into a
−40.46 EUR realised loss**, entirely through truncation.

A policy trained on realised bills can learn to only plan reductions it will be
able to deliver. That is a genuine mechanism, not a modelling trick, and it is
why this comparison is worth running.

---

## Honest limits, stated before the results

- **There is no four-year demand history.** The site file spans 3.05 years but
  only **494 days** have PV, EMOB1 and EMOB2 all complete (EMOB2 is populated
  44 % of the time); 487 survive the forecast lookback. The four years are on
  the **price** side, where they drive arbitrage value. The environment pairs
  demand days with price days by sampling, which turns 487 days into hundreds
  of thousands of training scenarios and forces the policy to answer the price
  *shape* rather than memorise a date.
- **The site split is estimated, not assumed.** Production splits the EV
  forecast by charger capacity (0.403 / 0.597); observed consumption splits
  0.723 / 0.277. That mismatch is a real defect in the deployed system and
  belongs in the deployment audit — but here it would make every MPC arm look
  bad for a reason unrelated to control, so the split is estimated **from the
  training days only**.
- **A baseline still missing.** `mpc_robust` — an MPC that sizes reductions
  against a lower quantile of demand rather than the point forecast — is the
  obvious fix for the truncation problem, and it must be added before claiming
  a learned policy beats "the MPC". Its quantile must be fitted on training
  days only. Without it, an RL win is partly a win over a known-fixable
  weakness, and saying otherwise would be overclaiming.
- Held-out days are the **last** ones chronologically. No leakage.

---

## Layout

| file | what it does |
|---|---|
| `fetch_prices_4y.py` | four years of DE-LU day-ahead prices, in yearly chunks |
| `build_dataset.py` | every complete day → `rl/data/*.npz`, with the forecast the controller would have had |
| `site_env.py` | the day as a closed-loop control problem; constraints enforced analytically |
| `baselines.py` | `mpc_deployed`, `mpc_ledger`, `mpc_rh`, `oracle` |
| `sac.py` | Soft Actor-Critic, written out rather than imported |
| `train.py` | training loop; checkpoints on **held-out** margin, never training reward |
| `evaluate.py` | the final paired comparison → `Data/results/e24_rl_project.csv` |

Feasibility is enforced, not learned: every action is clipped into the
deployed controller's exact admissible set — same ±200 kW bound, same 80 kW
ramp, same charger ceilings, same `sum(u) = 0`. The closure clip is ramp-aware,
reserving exactly the capacity still needed to return the running budget to
zero. The agent cannot win by cheating a constraint, and its setpoints are
deployable as they stand.

---

## Running it

From the repository root, with `PYTHONPATH=.`:

```bash
# 1. four years of prices  (needs your ENTSO-E token; rotate the old one first)
export ENTSOE_TOKEN=...
python -m rl.fetch_prices_4y

# 2. build the dataset
python -m rl.build_dataset

# 3. train
python -m rl.train --steps 600000 --seed 0

# 4. the answer
python -m rl.evaluate --ckpt rl/runs/sac_s0.pt
```

Step 3 prints a held-out comparison against `mpc_ledger` every 20 000 steps, so
you can see whether it is learning without waiting for the end. CPU only —
this machine has no CUDA, and the networks are small enough that it does not
matter.

**Run at least three seeds** (`--seed 0`, `1`, `2`) before believing any
margin. A single seed on 120 held-out days cannot distinguish a real 1 % effect
from noise: the day-to-day spread is ~18 % of daily cost, which is why every
number here is paired and reported with Wilcoxon and paired `d_z`.
