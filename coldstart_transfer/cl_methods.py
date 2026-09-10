"""
cl_methods.py -- additional continual-learning baselines beyond EWC/replay, to
answer the "missing baselines" critique. All warm-start from the production model
theta* and are evaluated with the same target/source-retention protocol.

  MAS   : regularisation, importance = mean|grad(0.5||f||^2)| (unsupervised)
  LwF   : distillation, keep teacher(theta*) outputs on TARGET inputs (no source data)
  DER++ : replay of source samples + distillation of the teacher's logits on them
  A-GEM : replay memory used only to PROJECT the target gradient (no source loss term)

Importance-based penalties use unit-mean normalisation (the scale-fairness control
from RQ3), so lambda's effective strength is comparable across methods.
"""
from __future__ import annotations
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model import build_ev_model, load_production_weights
from . import ewc as ewc_mod
from .trainer import nrmse, _normalize_fisher
from .windowing import first_n_days


def run_cl_method(method, target_train, target_holdout, scalers, prod_keras_path,
                  n_days=30, seed=0, epochs=10, batch_size=64, lr=1e-4,
                  lam=100.0, lam_lwf=1.0, der_alpha=0.5, der_beta=1.0,
                  source_train=None, source_holdout=None, m_imp=500,
                  model_builder=None):
    keras.utils.set_random_seed(seed)
    t0 = time.time()
    builder = model_builder or build_ev_model
    Xs, Xn, y, _ = first_n_days(*target_train, n_days=n_days)
    n = len(y); batch_size = min(batch_size, max(1, n))

    model = builder(seed=seed)
    load_production_weights(model, prod_keras_path)
    tvars = model.trainable_variables
    star = [tf.constant(v.numpy()) for v in tvars]

    needs_src = method in ("DERpp", "AGEM")
    if needs_src and source_train is None:
        raise ValueError(f"{method} needs source_train")

    teacher = None
    if method in ("LwF", "DERpp"):
        teacher = builder(); load_production_weights(teacher, prod_keras_path)
        teacher.trainable = False

    importance = None
    if method == "MAS":
        importance = ewc_mod.estimate_mas_importance(model, Xs, Xn, m_points=min(m_imp, n),
                                                     batch_size=batch_size, seed=seed)
        importance = _normalize_fisher(importance, mode="global")

    opt = keras.optimizers.Adam(lr)
    rng = np.random.default_rng(seed)

    def src_batch():
        m = len(source_train[2]); idx = rng.choice(m, size=min(batch_size, m), replace=False)
        return (tf.constant(source_train[0][idx]), tf.constant(source_train[1][idx]),
                tf.constant(source_train[2][idx]))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            bs = tf.constant(Xs[idx]); bn = tf.constant(Xn[idx]); by = tf.constant(y[idx])
            with tf.GradientTape() as tape:
                yhat = model([bs, bn], training=True)
                loss = ewc_mod.huber(by, yhat)
                if method == "MAS":
                    # ewc_penalty already includes the 1/2 factor -> lambda * penalty
                    loss += lam * ewc_mod.ewc_penalty(model, importance, star)
                if method == "LwF":
                    loss += lam_lwf * tf.reduce_mean(tf.square(
                        model([bs, bn], training=True) - teacher([bs, bn], training=False)))
                if method == "DERpp":
                    rs, rn, ry = src_batch()
                    tlog = teacher([rs, rn], training=False)
                    loss += der_alpha * tf.reduce_mean(tf.square(model([rs, rn], training=True) - tlog))
                    loss += der_beta * ewc_mod.huber(ry, model([rs, rn], training=True))
            grads = tape.gradient(loss, tvars)
            if method == "AGEM":
                rs, rn, ry = src_batch()
                with tf.GradientTape() as gt:
                    gl = ewc_mod.huber(ry, model([rs, rn], training=True))
                gref = gt.gradient(gl, tvars)
                dot = tf.add_n([tf.reduce_sum(g*gr) for g, gr in zip(grads, gref) if g is not None and gr is not None])
                if dot < 0:
                    ref2 = tf.add_n([tf.reduce_sum(gr*gr) for gr in gref if gr is not None]) + 1e-12
                    grads = [g - (dot/ref2)*gr if (g is not None and gr is not None) else g
                             for g, gr in zip(grads, gref)]
            opt.apply_gradients(zip(grads, tvars))

    tgt_nrmse, tgt_rmse, tgt_mean = nrmse(model, target_holdout, scalers)
    src_nrmse = nrmse(model, source_holdout, scalers)[0] if source_holdout is not None else None
    return {"baseline": method, "n_days": n_days, "lambda_ewc": lam, "beta_replay": der_beta,
            "seed": seed, "target_nrmse": round(tgt_nrmse, 6), "target_rmse_kw": round(tgt_rmse, 4),
            "target_mean_kw": round(tgt_mean, 4),
            "source_retention_nrmse": round(src_nrmse, 6) if src_nrmse is not None else "",
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "wall_clock_s": round(time.time()-t0, 1),
            "model": model}   # for optional reuse; logger drops unknown fields
