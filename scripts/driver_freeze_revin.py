"""
driver_freeze_revin.py -- freeze-depth sweep in the RevIN representation.

WHY: the whole CL benchmark (rq1/rq2/ewc_bench/cl_methods/freeze_sweep) runs on the
FROZEN production MinMax scalers. The representation ablation shows that is the
WORST representation for this transfer (B3 = 0.95 nRMSE) while RevIN reaches 0.116
-- 2.8x better than the naive-1 persistence baseline (0.324). So every CL conclusion
so far is drawn in the regime where the model performs worst.

This asks whether the freeze-depth result survives a well-calibrated model:
does freezing the sequence trunk still dominate full fine-tuning under RevIN?

Both outcomes are informative:
  - freeze3 still wins  -> the result is representation-robust.
  - freeze3 vanishes    -> the representation already absorbs what freezing fixed,
                           which is the strongest possible form of the paper's thesis.

Protocol mirrors driver_representation_ablation.py (same windows, same RevIN, same
source-model recipe: 12 epochs @1e-3, then N=30 target fine-tune 10 epochs @1e-4)
and adds the source-retention probe of driver_source.py.

Writes Data/results/freeze_revin.csv. Touches no existing result file.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Reuse the ablation's exact windowing + representation (importing it also fits the
# shared exogenous MinMax scaler on the source, as the ablation does).
from scripts.driver_representation_ablation import raw_windows, RevINRep, LB, src, tgt
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.logger import log_result

N_DAYS = int(os.environ.get("N_DAYS", "30"))
HOLD = int(os.environ.get("HOLD", "14"))          # target holdout days
RET_DAYS = int(os.environ.get("RET_DAYS", "7"))   # source retention holdout days
SRC_DAYS = int(os.environ.get("SRC_DAYS", "150"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
LEVELS = [int(x) for x in os.environ.get("LEVELS", "0,1,2,3,4,5").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-4"))
SRC_EPOCHS = int(os.environ.get("SRC_EPOCHS", "12"))
BETA = float(os.environ.get("BETA", "1.0"))
WITH_REPLAY_REF = os.environ.get("WITH_REPLAY_REF", "1") == "1"
OUT = os.environ.get("OUT", "Data/results/freeze_revin.csv")

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def ev_groups(model):
    """Same trunk-ordered freeze groups as driver_freeze_sweep.py."""
    convs = [l for l in model.layers if isinstance(l, keras.layers.Conv1D)]
    lstms = [l for l in model.layers if isinstance(l, keras.layers.LSTM)]
    denses = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    units = [l.units for l in denses]
    assert len(convs) == 2 and len(lstms) == 1, "unexpected trunk"
    assert units == [64, 128, 1], f"unexpected dense units: {units}"
    return [convs[0], convs[1], lstms[0], denses[0], denses[1]]


def apply_freeze(model, level):
    if level <= 0:
        return 0
    n = 0
    for lyr in ev_groups(model)[:level]:
        lyr.trainable = False
        n += int(sum(int(w.shape.num_elements()) for w in lyr.weights))
    return n


def nrmse_kw(model, Xs, Xn, denorm, y_true_kw):
    pred = model.predict([Xs, Xn], verbose=0).reshape(-1)
    pred_kw = np.clip(denorm(pred), 0, None)
    rmse = float(np.sqrt(np.mean((pred_kw - y_true_kw) ** 2)))
    return rmse / (float(y_true_kw.mean()) + 1e-9)


def huber(y, yhat):
    return tf.reduce_mean(keras.losses.huber(tf.reshape(y, (-1, 1)),
                                             tf.reshape(yhat, (-1, 1)), delta=0.5))


def train(model, Xs, Xn, y, seed, replay=None):
    """Fine-tune loop mirroring trainer.run_condition (optional source replay)."""
    opt = keras.optimizers.Adam(LR)
    tvars = model.trainable_variables
    rng = np.random.default_rng(seed)
    n = len(y)
    bs_ = min(BATCH, n)
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, bs_):
            idx = order[s:s + bs_]
            bx, bn, by = tf.constant(Xs[idx]), tf.constant(Xn[idx]), tf.constant(y[idx])
            with tf.GradientTape() as tape:
                loss = huber(by, model([bx, bn], training=True))
                if replay is not None:
                    m = len(replay[2])
                    ri = rng.choice(m, size=min(bs_, m), replace=False)
                    loss = loss + BETA * huber(
                        tf.constant(replay[2][ri]),
                        model([tf.constant(replay[0][ri]), tf.constant(replay[1][ri])],
                              training=True))
            opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
    return model


def main():
    rep = RevINRep()

    # ---- SOURCE windows: train span + a final RET_DAYS retention holdout ----
    src_tail = src.iloc[-(SRC_DAYS * 96 + LB):].reset_index(drop=True)
    sXs, sXn, sy, _ = raw_windows(src_tail)
    cut = RET_DAYS * 96
    s_tr_raw = (sXs[:-cut], sXn[:-cut], sy[:-cut])
    s_ho_raw = (sXs[-cut:], sXn[-cut:], sy[-cut:])
    sXs_t, sXn_t, sy_t, _ = rep.apply(*s_tr_raw)
    sho_Xs, sho_Xn, _, s_denorm_ho = rep.apply(*s_ho_raw)
    s_ho_kw = s_ho_raw[2]

    # ---- TARGET windows: first N_DAYS for training, last HOLD days for holdout ----
    tgt_head = tgt.iloc[:LB + N_DAYS * 96].reset_index(drop=True)
    thXs, thXn, thy, _ = raw_windows(tgt_head)
    hXs_r, hXn_r, hy_r, _ = raw_windows(tgt.iloc[-(LB + HOLD * 96):].reset_index(drop=True))
    tXs_t, tXn_t, ty_t, _ = rep.apply(thXs, thXn, thy)
    tho_Xs, tho_Xn, _, t_denorm_ho = rep.apply(hXs_r, hXn_r, hy_r)
    t_ho_kw = hy_r

    print(f"[revin] source train={len(sy_t)} retention_holdout={len(s_ho_kw)} | "
          f"target train={len(ty_t)} holdout={len(t_ho_kw)}", flush=True)
    print(f"[revin] levels={LEVELS} seeds={SEEDS} replay_ref={WITH_REPLAY_REF}", flush=True)

    # ---- source model, trained ONCE in this representation (ablation recipe) ----
    keras.utils.set_random_seed(0)
    srcm = build_ev_model(seed=0)
    srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
    srcm.fit([sXs_t, sXn_t], sy_t, epochs=SRC_EPOCHS, batch_size=128, verbose=0)
    src_w = srcm.get_weights()
    print(f"[revin] source model trained | source-holdout nRMSE="
          f"{nrmse_kw(srcm, sho_Xs, sho_Xn, s_denorm_ho, s_ho_kw):.4f}", flush=True)

    replay_pool = (sXs_t, sXn_t, sy_t)

    def run(level, seed, use_replay, tag):
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        m = build_ev_model(seed=seed)
        m.set_weights(src_w)
        nf = apply_freeze(m, level)
        train(m, tXs_t, tXn_t, ty_t, seed,
              replay=replay_pool if use_replay else None)
        t = nrmse_kw(m, tho_Xs, tho_Xn, t_denorm_ho, t_ho_kw)
        s = nrmse_kw(m, sho_Xs, sho_Xn, s_denorm_ho, s_ho_kw)
        log_result(OUT, dict(baseline="B5" if use_replay else "B3", n_days=N_DAYS,
                             lambda_ewc=0.0, beta_replay=BETA if use_replay else 0.0,
                             seed=seed, target_nrmse=round(t, 6),
                             source_retention_nrmse=round(s, 6),
                             epochs=EPOCHS, batch_size=BATCH, lr=LR, notes=tag))
        print(f"[revin {tag} s={seed}] target={t:.4f} retention={s:.4f} frozen={nf}",
              flush=True)

    for seed in SEEDS:
        for lvl in LEVELS:
            run(lvl, seed, False, f"freeze{lvl}")
        if WITH_REPLAY_REF:
            run(0, seed, True, "replay_ref")

    print(f"[revin] DONE -> {OUT}")


if __name__ == "__main__":
    main()
