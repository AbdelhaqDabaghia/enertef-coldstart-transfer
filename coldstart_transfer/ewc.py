"""
ewc.py -- Elastic Weight Consolidation with the PRODUCTION output-sensitivity
Fisher proxy (NOT the textbook per-example empirical Fisher), reused verbatim
from the Luxembourg nightly system:

    F_ii = (1/B) * sum_b [ d/dtheta_i ( (1/|batch_b|) * sum_j y_hat_j^2 ) ]^2

estimated over B mini-batches drawn from M sampled points (M=500 in production;
fewer if the sample has fewer points, e.g. B4 target re-estimation with < 500).

EWC objective (same mathematical form as production):
    L_EWC(theta) = L_data + (lambda/2) * sum_i F_ii * (theta_i - theta_A_i*)^2

MODES (same estimator, different data):
  - source-transfer (B1): Fisher estimated on LUXEMBOURG source windows.
  - target-re-estimation (B4): Fisher estimated on the TARGET N-day windows,
    but model INPUT scaling still uses the frozen LU scalers (weight compat).
The star params theta_A* are the transferred production weights (warm start).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


def estimate_fisher(model, X_seq, X_next, m_points=500, batch_size=32,
                    seed=0):
    """Diagonal Fisher via the output-sensitivity proxy. Returns a list of
    tf.Tensors, one per model.trainable_variables, same shapes.

    Samples min(m_points, N) points, splits them into B mini-batches of
    batch_size, and for each batch accumulates the squared gradient of the
    batch-mean squared output w.r.t. each parameter."""
    n = len(X_seq)
    m = min(m_points, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    xs = tf.constant(X_seq[idx]); xn = tf.constant(X_next[idx])

    tvars = model.trainable_variables
    fisher = [tf.zeros_like(v) for v in tvars]
    n_batches = 0
    for start in range(0, m, batch_size):
        bs, bn = xs[start:start + batch_size], xn[start:start + batch_size]
        if bs.shape[0] == 0:
            continue
        with tf.GradientTape() as tape:
            y_hat = model([bs, bn], training=False)
            # batch-mean squared output (the production sensitivity signal)
            signal = tf.reduce_mean(tf.square(y_hat))
        grads = tape.gradient(signal, tvars)
        for i, g in enumerate(grads):
            if g is not None:
                fisher[i] += tf.square(g)   # [ d signal / d theta_i ]^2
        n_batches += 1
    if n_batches == 0:
        raise ValueError("no batches formed for Fisher estimation")
    fisher = [f / float(n_batches) for f in fisher]
    return fisher


def estimate_empirical_fisher(model, X_seq, X_next, y, m_points=200, batch_size=1,
                              seed=0, delta=0.5):
    """TRUE (empirical) diagonal Fisher: average of the squared gradient of the
    per-example Huber LOSS w.r.t. each parameter, using the real targets y.
    Contrasts with estimate_fisher()'s output-sensitivity proxy (which ignores y).
    Per-example (batch_size=1) is the honest estimator; m_points kept modest since
    it costs m backward passes."""
    n = len(X_seq)
    m = min(m_points, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    tvars = model.trainable_variables
    fisher = [tf.zeros_like(v) for v in tvars]
    for i in idx:
        xs = tf.constant(X_seq[i:i+1]); xn = tf.constant(X_next[i:i+1])
        yi = tf.constant(y[i:i+1])
        with tf.GradientTape() as tape:
            yhat = model([xs, xn], training=False)
            loss = huber(yi, yhat, delta)
        for k, g in enumerate(tape.gradient(loss, tvars)):
            if g is not None:
                fisher[k] += tf.square(g)
    return [f / float(m) for f in fisher]


def estimate_mas_importance(model, X_seq, X_next, m_points=500, batch_size=32, seed=0):
    """Memory Aware Synapses importance: mean over sampled points of the ABSOLUTE
    gradient of 0.5*||f(x)||^2 w.r.t. each parameter (unsquared, unlike Fisher).
    MAS is unsupervised (no labels), like our output-sensitivity proxy but abs
    rather than squared -- included to cover the full regularisation family."""
    n = len(X_seq); m = min(m_points, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    xs = tf.constant(X_seq[idx]); xn = tf.constant(X_next[idx])
    tvars = model.trainable_variables
    imp = [tf.zeros_like(v) for v in tvars]
    nb = 0
    for s in range(0, m, batch_size):
        bs, bn = xs[s:s+batch_size], xn[s:s+batch_size]
        if bs.shape[0] == 0:
            continue
        with tf.GradientTape() as tape:
            y = model([bs, bn], training=False)
            l2 = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(y), axis=-1))
        for i, g in enumerate(tape.gradient(l2, tvars)):
            if g is not None:
                imp[i] += tf.abs(g)
        nb += 1
    return [w / float(max(nb, 1)) for w in imp]


def ewc_penalty(model, fisher, star_params):
    """(1/2) * sum_i F_ii * (theta_i - theta_i*)^2  (lambda applied by caller)."""
    total = tf.constant(0.0)
    for v, f, star in zip(model.trainable_variables, fisher, star_params):
        total += tf.reduce_sum(f * tf.square(v - star))
    return 0.5 * total


def snapshot_params(model):
    """theta_A* : copy of current (warm-started) weights, as constants."""
    return [tf.constant(v.numpy()) for v in model.trainable_variables]


def huber(y_true, y_pred, delta=0.5):
    """Huber loss, delta=0.5 (production)."""
    err = y_true - tf.reshape(y_pred, [-1])
    a = tf.abs(err)
    quad = tf.minimum(a, delta)
    lin = a - quad
    return tf.reduce_mean(0.5 * quad ** 2 + delta * lin)


@tf.function(reduce_retracing=True)
def _noop():
    return None
