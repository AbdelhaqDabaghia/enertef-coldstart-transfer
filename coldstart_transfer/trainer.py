"""
trainer.py -- single parameterised training/evaluation loop implementing all
six cold-start conditions (B0-B5) as config variants of ONE core loop.

    B0  from-scratch      : random init (seeded), no warm-start, plain Huber
    B1  warm + EWC(source): warm-start prod weights, EWC with SOURCE Fisher
    B2  naive fine-tune   : warm-start, no EWC (== lambda 0)   [RQ2 anchor]
    B3  warm, no EWC      : warm-start, plain fine-tune         [RQ1 init-only]
    B4  warm + EWC(target): warm-start, EWC with TARGET-re-estimated Fisher
    B5  warm + replay     : warm-start, replay SOURCE batches mixed in

Loss (superset):  L = Huber(data) + (lambda/2)*EWC_penalty + beta*Huber(replay)

All conditions use the FROZEN production scalers (windowing.py). B4 re-estimates
Fisher on TARGET windows but still scales inputs with the LU scalers.

Runnable WITHOUT source data: B0, B2, B3, B4 (B4's Fisher is from target).
Need source features: B1 (source Fisher), B5 (replay), + source-retention metric.
"""
from __future__ import annotations
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model import build_ev_model, load_production_weights
from . import ewc as ewc_mod
from .windowing import inverse_target

# baseline -> behaviour flags
_SPEC = {
    "B0": dict(warm=False, use_ewc=False, fisher="none",   replay=False),
    "B1": dict(warm=True,  use_ewc=True,  fisher="source", replay=False),
    "B2": dict(warm=True,  use_ewc=False, fisher="none",   replay=False),
    "B3": dict(warm=True,  use_ewc=False, fisher="none",   replay=False),
    "B4": dict(warm=True,  use_ewc=True,  fisher="target", replay=False),
    "B5": dict(warm=True,  use_ewc=False, fisher="none",   replay=True),
    # --- honest improved-EWC variants (bench) ---
    "V1": dict(warm=True,  use_ewc=True,  fisher="empirical_target", replay=False),
    "V2": dict(warm=True,  use_ewc=True,  fisher="target",           replay=False),
    "V3": dict(warm=True,  use_ewc=True,  fisher="empirical_target", replay=True),
}


def nrmse(model, holdout, scalers):
    """nRMSE = RMSE / mean(true power), computed in kW (un-scaled), non-neg preds."""
    Xs, Xn, y, _ = holdout
    pred_s = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
    pred_kw = inverse_target(pred_s, scalers)
    true_kw = inverse_target(y, scalers)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2)))
    mean = float(np.mean(true_kw))
    return (rmse / mean if mean > 1e-9 else float("nan")), rmse, mean


def _normalize_fisher(fisher, mode="global"):
    """Rescale the diagonal Fisher so lambda's effective strength is comparable.
      global  : divide everything by the global mean (scale -> O(1)).
      perlayer: divide EACH tensor by its own mean, so no single layer's Fisher
                dominates the penalty (lets layers that must adapt stay free)."""
    if mode == "perlayer":
        return [f / (tf.reduce_mean(f) + 1e-12) for f in fisher]
    total = tf.add_n([tf.reduce_sum(f) for f in fisher])
    count = tf.add_n([tf.cast(tf.size(f), tf.float32) for f in fisher])
    return [f / (total / count + 1e-12) for f in fisher]


def run_condition(baseline, target_train, target_holdout, scalers,
                  prod_keras_path, n_days=30, lambda_ewc=0.0, beta_replay=0.0,
                  seed=0, epochs=10, batch_size=32, lr=1e-4,
                  source_train=None, source_holdout=None, m_fisher=500,
                  normalize_fisher=False, normalize_mode="global",
                  model_builder=None, freeze_fn=None):
    """Run one condition end-to-end; return a result dict (see logger.FIELDS).

    freeze_fn: optional callable(model) applied right after the warm-start, used
    by the freeze-depth sweep to set layer.trainable=False on part of the trunk.
    Left as None (the default) the code path is byte-for-byte the B0-B5 one."""
    spec = _SPEC[baseline]
    keras.utils.set_random_seed(seed)
    t0 = time.time()

    # --- data: first N days of the (holdout-excluded) target training stream ---
    from .windowing import first_n_days
    Xs, Xn, y, _ = first_n_days(*target_train, n_days=n_days)
    n = len(y)
    if n < batch_size:
        batch_size = max(1, n)

    # --- model: warm-start or from-scratch (EV builder by default; pass
    #     model_builder=build_pv_model for the PV replication) ---
    builder = model_builder or build_ev_model
    model = builder(seed=seed)
    if spec["warm"]:
        load_production_weights(model, prod_keras_path)
    # Freeze BEFORE the Fisher/star snapshot so every downstream use of
    # model.trainable_variables sees the same, already-reduced variable list.
    n_frozen = freeze_fn(model) if freeze_fn is not None else 0
    star = ewc_mod.snapshot_params(model) if spec["use_ewc"] else None

    # --- Fisher (source-transfer B1, or target-re-estimation B4) ---
    fisher = None
    if spec["use_ewc"]:
        if spec["fisher"] == "source":
            if source_train is None:
                raise ValueError("B1 needs source_train for source Fisher")
            fisher = ewc_mod.estimate_fisher(model, source_train[0], source_train[1],
                                             m_points=m_fisher, batch_size=batch_size)
        elif spec["fisher"] == "empirical_target":
            fisher = ewc_mod.estimate_empirical_fisher(
                model, Xs, Xn, y, m_points=min(m_fisher, n), seed=seed)
        else:  # proxy target
            fisher = ewc_mod.estimate_fisher(model, Xs, Xn,
                                             m_points=min(m_fisher, n),
                                             batch_size=batch_size)
        if normalize_fisher:
            fisher = _normalize_fisher(fisher, mode=normalize_mode)

    # --- replay source pool (B5) ---
    if spec["replay"] and source_train is None:
        raise ValueError("B5 needs source_train for replay")

    opt = keras.optimizers.Adam(lr)
    tvars = model.trainable_variables
    rng = np.random.default_rng(seed)

    def batch_loss(bs, bn, by, training):
        yhat = model([bs, bn], training=training)
        loss = ewc_mod.huber(by, yhat)
        if spec["replay"]:
            m = len(source_train[2])
            ri = rng.choice(m, size=min(batch_size, m), replace=False)
            rs = tf.constant(source_train[0][ri]); rn = tf.constant(source_train[1][ri])
            ry = tf.constant(source_train[2][ri])
            loss = loss + beta_replay * ewc_mod.huber(ry, model([rs, rn], training=training))
        if spec["use_ewc"] and lambda_ewc > 0:
            loss = loss + lambda_ewc * ewc_mod.ewc_penalty(model, fisher, star)
        return loss

    # --- training loop ---
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            bs = tf.constant(Xs[idx]); bn = tf.constant(Xn[idx]); by = tf.constant(y[idx])
            with tf.GradientTape() as tape:
                loss = batch_loss(bs, bn, by, training=True)
            opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))

    # --- evaluation ---
    tgt_nrmse, tgt_rmse, tgt_mean = nrmse(model, target_holdout, scalers)
    src_nrmse = None
    if source_holdout is not None:
        src_nrmse, _, _ = nrmse(model, source_holdout, scalers)

    return {
        "baseline": baseline, "n_days": n_days, "lambda_ewc": lambda_ewc,
        "beta_replay": beta_replay, "seed": seed,
        "target_nrmse": round(tgt_nrmse, 6), "target_rmse_kw": round(tgt_rmse, 4),
        "target_mean_kw": round(tgt_mean, 4),
        "source_retention_nrmse": round(src_nrmse, 6) if src_nrmse is not None else "",
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "n_frozen_params": n_frozen,
        "wall_clock_s": round(time.time() - t0, 1),
        "model": model,  # returned for optional reuse; not logged
    }
