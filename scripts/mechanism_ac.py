"""
Mechanism (a) + (c).

(a) Weights stay near theta* under EWC yet target error does not improve: we log,
    per epoch, the relative displacement ||theta-theta*||/||theta*|| and the target
    error, for plain fine-tune (B3) vs source-Fisher EWC (B1). EWC keeps theta
    close to theta* (small displacement) but pays HIGHER target error -- anchoring
    to the source optimum is the problem, because (mechanism b) the target must
    move exactly the anchored directions.

(c) Replay quality depends on COVERAGE of the old data: we sweep the replay buffer
    size and measure source retention. Retention should improve with coverage,
    showing replay's advantage is a direct re-exposure effect (and quantifying the
    memory budget it needs).
"""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days, inverse_target
from coldstart_transfer import ewc as ewc_mod

sc = joblib.load("Data/models/ev_scalers.joblib"); PROD = "Data/models/ev_cnn_lstm_20260718.keras"
tgt = pd.read_csv("Data/uk_ev_features_full.csv"); tX, tN, ty, tts = make_windows(tgt, sc)
ttr, tho = split_holdout(tX, tN, ty, tts, 14); xs, xn, yy, _ = first_n_days(*ttr, n_days=30)
src = pd.read_csv("Data/lux_source_features.csv"); tail=(60+7)*96+672
sX, sN, sy, sts = make_windows(src.iloc[-tail:].reset_index(drop=True), sc)
spool, shold = split_holdout(sX, sN, sy, sts, 7)

def nrmse(m, H):
    p = np.clip(m.predict([H[0], H[1]], verbose=0).reshape(-1), 0, None)
    pk = inverse_target(p, sc); tk = inverse_target(H[2], sc)
    return float(np.sqrt(np.mean((pk-tk)**2)))/float(np.mean(tk))

def disp(m, star):
    num = np.sqrt(sum(float(np.sum((v.numpy()-s.numpy())**2)) for v,s in zip(m.trainable_variables, star)))
    den = np.sqrt(sum(float(np.sum(s.numpy()**2)) for s in star)); return num/den

print("=== (a) displacement vs error ===")
for method in ("B3_plain", "B1_ewc"):
    keras.utils.set_random_seed(0); m = build_ev_model(seed=0); load_production_weights(m, PROD)
    star = ewc_mod.snapshot_params(m); tv = m.trainable_variables
    F = ewc_mod.estimate_fisher(m, spool[0], spool[1], m_points=500, batch_size=64) if method=="B1_ewc" else None
    if F is not None:
        mean = tf.add_n([tf.reduce_sum(f) for f in F])/tf.add_n([tf.cast(tf.size(f),tf.float32) for f in F]); F=[f/(mean+1e-12) for f in F]
    opt = keras.optimizers.Adam(1e-4); rng = np.random.default_rng(0); n=len(yy)
    for ep in range(10):
        for s in range(0, n, 64):
            i = rng.permutation(n)[s:s+64]
            with tf.GradientTape() as tp:
                loss = ewc_mod.huber(tf.constant(yy[i]), m([tf.constant(xs[i]), tf.constant(xn[i])], training=True))
                if F is not None: loss += 100.0*ewc_mod.ewc_penalty(m, F, star)
            opt.apply_gradients(zip(tp.gradient(loss, tv), tv))
        if ep in (0,2,5,9):
            print(f"  [{method} ep{ep+1}] disp={disp(m,star):.4f} target_nRMSE={nrmse(m,tho):.4f} src_ret={nrmse(m,shold):.4f}")

print("=== (c) replay retention vs buffer coverage ===")
for B in (50, 200, 1000, len(spool[2])):
    keras.utils.set_random_seed(0); m = build_ev_model(seed=0); load_production_weights(m, PROD)
    tv = m.trainable_variables; opt = keras.optimizers.Adam(1e-4); rng = np.random.default_rng(0); n=len(yy)
    buf = rng.choice(len(spool[2]), size=min(B, len(spool[2])), replace=False)
    for ep in range(10):
        for s in range(0, n, 64):
            i = rng.permutation(n)[s:s+64]; ri = rng.choice(buf, size=min(64, len(buf)), replace=False)
            with tf.GradientTape() as tp:
                loss = ewc_mod.huber(tf.constant(yy[i]), m([tf.constant(xs[i]), tf.constant(xn[i])], training=True))
                loss += 1.0*ewc_mod.huber(tf.constant(spool[2][ri]), m([tf.constant(spool[0][ri]), tf.constant(spool[1][ri])], training=True))
            opt.apply_gradients(zip(tp.gradient(loss, tv), tv))
    print(f"  [replay buffer={B:5d}] src_retention_nRMSE={nrmse(m,shold):.4f} target_nRMSE={nrmse(m,tho):.4f}")
print("DONE")
