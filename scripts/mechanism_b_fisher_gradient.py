"""
Mechanism (b): do the Fisher-important directions still match the USEFUL target
gradients? At theta* (production model), we compare the source Fisher importance
F_i to the magnitude |g_i| of the target-loss gradient. If EWC protects (high F)
the very parameters the target needs to move (high |g|), it blocks useful
adaptation -- the mechanistic reason EWC hurts plasticity across regimes.
"""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from scipy.stats import spearmanr
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days
from coldstart_transfer import ewc as ewc_mod

sc = joblib.load("Data/models/ev_scalers.joblib")
PROD = "Data/models/ev_cnn_lstm_20260718.keras"
m = build_ev_model(); load_production_weights(m, PROD)
tvars = m.trainable_variables

# source Fisher (importance EWC protects)
src = pd.read_csv("Data/lux_source_features.csv"); tail = (60+7)*96+672
sX, sN, sy, sts = make_windows(src.iloc[-tail:].reset_index(drop=True), sc)
spool, _ = split_holdout(sX, sN, sy, sts, 7)
F = ewc_mod.estimate_fisher(m, spool[0], spool[1], m_points=500, batch_size=64)

# target gradient (the useful adaptation direction), averaged over target N=30
tgt = pd.read_csv("Data/uk_ev_features_full.csv")
tX, tN, ty, tts = make_windows(tgt, sc); tr, _ = split_holdout(tX, tN, ty, tts, 14)
xs, xn, yy, _ = first_n_days(*tr, n_days=30)
idx = np.random.default_rng(0).choice(len(yy), 500, replace=False)
with tf.GradientTape() as tape:
    yhat = m([tf.constant(xs[idx]), tf.constant(xn[idx])], training=False)
    loss = ewc_mod.huber(tf.constant(yy[idx]), yhat)
G = [abs(g.numpy()) for g in tape.gradient(loss, tvars)]

f = np.concatenate([x.numpy().ravel() for x in F]); g = np.concatenate([x.ravel() for x in G])
f = f / (f.sum() + 1e-12); g = g / (g.sum() + 1e-12)     # normalise to distributions
rho, p = spearmanr(f, g)
cos = float(np.dot(f, g) / (np.linalg.norm(f)*np.linalg.norm(g) + 1e-12))
# how much of the target-gradient ENERGY sits in the top-decile Fisher params?
thr = np.quantile(f, 0.90); frac_top = float(g[f >= thr].sum())
print(f"[mech-b] Spearman(Fisher, |target grad|) = {rho:+.3f} (p={p:.1e})")
print(f"[mech-b] cosine(Fisher, |target grad|)    = {cos:+.3f}")
print(f"[mech-b] target-gradient mass in top-10% Fisher dirs = {frac_top:.1%} "
      f"(10% if unrelated; >10% => EWC blocks useful updates)")
print(f"[mech-b] target-gradient mass in BOTTOM-90% Fisher dirs = {1-frac_top:.1%}")
