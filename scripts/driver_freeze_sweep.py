"""
driver_freeze_sweep.py -- freeze-depth sweep on the EV task (LU -> UK), N=30.

WHY: contention_profile.py shows the source-Fisher/target-gradient overlap is
NOT uniform across the network. On EV the sequence trunk is heavily contended
(enrich10 4-9x chance) while the dense head is BELOW chance (0.26-0.53x): the
target gradient there avoids the parameters the source relies on. Every B0-B5
baseline fine-tunes the whole network, so this axis is untested.

Freezing the trunk preserves the source representation BY CONSTRUCTION -- no
buffer, no penalty -- at the cost of some plasticity. This sweep measures that
trade-off and asks whether any freeze depth dominates B3 (warm-start) or B5
(replay) on the stability-plasticity plane.

Conditions: B3 at freeze levels 0..5 (cumulative, ordered by trunk depth), plus
B5 at level 0 as the replay reference. Level 0 == plain B3, i.e. the sweep
contains its own control.

  L0  nothing frozen            (== B3)
  L1  + conv1                   (enrich10 ~4.0)
  L2  + conv2                   (enrich10 ~7.8)
  L3  + LSTM                    (enrich10 ~9.0)  <- whole sequence encoder
  L4  + dense(64) next-branch   (enrich10 ~2.1)
  L5  + dense(128) head         (enrich10 ~0.45) -> only the output unit adapts

Writes Data/results/freeze_sweep.csv. Touches no existing result file.
Run: SEEDS=0,1,..,9 python -m scripts.driver_freeze_sweep
"""
from __future__ import annotations
import os
import joblib
import pandas as pd
from tensorflow import keras

from coldstart_transfer.windowing import make_windows, split_holdout
from coldstart_transfer.trainer import run_condition
from coldstart_transfer.logger import log_result

PROD = "Data/models/ev_cnn_lstm_20260718.keras"
SCALERS = "Data/models/ev_scalers.joblib"
TARGET = "Data/uk_ev_features_full.csv"
SOURCE = "Data/lux_source_features.csv"
OUT = os.environ.get("OUT", "Data/results/freeze_sweep.csv")

# protocol constants: identical to driver_source.py so results are comparable
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LEVELS = [int(x) for x in os.environ.get("LEVELS", "0,1,2,3,4,5").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
BETA = float(os.environ.get("BETA", "1.0"))
POOL_DAYS = int(os.environ.get("POOL_DAYS", "60"))
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))
WITH_REPLAY_REF = os.environ.get("WITH_REPLAY_REF", "1") == "1"


def _ev_groups(model):
    """Trunk-ordered freeze groups for the EV CNN-LSTM.

    Asserts the architecture it expects, so a model change fails loudly here
    instead of silently freezing the wrong thing."""
    convs = [l for l in model.layers if isinstance(l, keras.layers.Conv1D)]
    lstms = [l for l in model.layers if isinstance(l, keras.layers.LSTM)]
    denses = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    units = [l.units for l in denses]
    assert len(convs) == 2 and len(lstms) == 1, f"unexpected trunk: {convs} {lstms}"
    assert units == [64, 128, 1], f"unexpected dense order/units: {units}"
    # cumulative order = increasing depth through the sequence trunk, then head
    return [convs[0], convs[1], lstms[0], denses[0], denses[1]]


def make_freeze_fn(level):
    """Return freeze_fn(model) -> n_frozen_params, freezing groups 1..level."""
    def freeze_fn(model):
        if level <= 0:
            return 0
        groups = _ev_groups(model)
        n = 0
        for lyr in groups[:level]:
            lyr.trainable = False
            n += int(sum(int(w.shape.num_elements()) for w in lyr.weights))
        return n
    return freeze_fn


def main():
    scalers = joblib.load(SCALERS)

    tdf = pd.read_csv(TARGET)
    Xs, Xn, y, ts = make_windows(tdf, scalers)
    target_train, target_holdout = split_holdout(Xs, Xn, y, ts, holdout_days=14)

    sdf = pd.read_csv(SOURCE)
    tail_rows = (POOL_DAYS + RET_DAYS) * 96 + 672
    sdf_tail = sdf.iloc[-tail_rows:].reset_index(drop=True)
    sXs, sXn, sy, sts = make_windows(sdf_tail, scalers)
    source_pool, source_holdout = split_holdout(sXs, sXn, sy, sts,
                                                holdout_days=RET_DAYS)
    print(f"[freeze] target train={len(target_train[2])} holdout={len(target_holdout[2])} | "
          f"source pool={len(source_pool[2])} retention_holdout={len(source_holdout[2])}")
    print(f"[freeze] N={N_DAYS} levels={LEVELS} seeds={SEEDS} replay_ref={WITH_REPLAY_REF}")

    for seed in SEEDS:
        for lvl in LEVELS:
            r = run_condition("B3", target_train, target_holdout, scalers, PROD,
                              n_days=N_DAYS, lambda_ewc=0.0, seed=seed,
                              epochs=EPOCHS, batch_size=BATCH, lr=LR,
                              source_train=source_pool,
                              source_holdout=source_holdout,
                              freeze_fn=make_freeze_fn(lvl))
            nf = r.pop("n_frozen_params", 0)
            r.pop("model", None)
            r["notes"] = f"freeze{lvl}"
            log_result(OUT, r)
            print(f"[freeze L{lvl} s={seed}] target={r['target_nrmse']:.4f} "
                  f"retention={r['source_retention_nrmse']:.4f} "
                  f"frozen={nf} t={r['wall_clock_s']}s")

        if WITH_REPLAY_REF:
            r = run_condition("B5", target_train, target_holdout, scalers, PROD,
                              n_days=N_DAYS, lambda_ewc=0.0, beta_replay=BETA,
                              seed=seed, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                              source_train=source_pool,
                              source_holdout=source_holdout)
            r.pop("n_frozen_params", 0)
            r.pop("model", None)
            r["notes"] = "replay_ref"
            log_result(OUT, r)
            print(f"[freeze REPLAY s={seed}] target={r['target_nrmse']:.4f} "
                  f"retention={r['source_retention_nrmse']:.4f} t={r['wall_clock_s']}s")

    print(f"[freeze] DONE -> {OUT}")


if __name__ == "__main__":
    main()
