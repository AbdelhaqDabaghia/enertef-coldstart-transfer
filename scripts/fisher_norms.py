"""Compare the norm/scale of the SOURCE Fisher (B1) vs the TARGET Fisher (B4).
If they differ by orders of magnitude, the same lambda means very different
effective regularization -> must normalize or lambda-sweep B1 before concluding.
"""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days
from coldstart_transfer import ewc

print("GPU:", tf.config.list_physical_devices('GPU'))
scalers = joblib.load("Data/models/ev_scalers.joblib")

tdf = pd.read_csv("Data/uk_ev_features_full.csv")
Xs, Xn, y, ts = make_windows(tdf, scalers)
ttr, _ = split_holdout(Xs, Xn, y, ts, 14)
tgt = first_n_days(*ttr, n_days=30)

sdf = pd.read_csv("Data/lux_source_features.csv")
tail = (60 + 7) * 96 + 672
sXs, sXn, sy, sts = make_windows(sdf.iloc[-tail:].reset_index(drop=True), scalers)
spool, _ = split_holdout(sXs, sXn, sy, sts, 7)

m = build_ev_model(); load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")
F_src = ewc.estimate_fisher(m, spool[0], spool[1], m_points=500, batch_size=64)
F_tgt = ewc.estimate_fisher(m, tgt[0], tgt[1], m_points=500, batch_size=64)

def stats(F):
    flat = np.concatenate([f.numpy().ravel() for f in F])
    return flat.sum(), flat.mean(), np.sqrt((flat**2).sum())

ss, sm, sl2 = stats(F_src)
ts_, tm, tl2 = stats(F_tgt)
print(f"\n{'Fisher':8s} {'sum':>12s} {'mean':>12s} {'L2':>12s}")
print(f"{'source':8s} {ss:12.4e} {sm:12.4e} {sl2:12.4e}")
print(f"{'target':8s} {ts_:12.4e} {tm:12.4e} {tl2:12.4e}")
print(f"\nratio source/target : sum={ss/ts_:.3f}  mean={sm/tm:.3f}  L2={sl2/tl2:.3f}")
