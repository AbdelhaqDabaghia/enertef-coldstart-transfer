"""
Sign-condition verification for the scale-decomposition model of EWC importance
mismatch (paper section "A scale-decomposition model ...").

We DO NOT retrain. Using the DEPLOYED production EV model, we estimate the diagonal
output-sensitivity Fisher importance Omega on four (site, representation) pairs and
form the scalar summary S(Omega) = mean over all diagonal entries:

  S_s_raw   : LU source windows, source-scaler pipeline        (raw representation)
  S_t_raw   : UK target windows, source-scaler pipeline        (raw representation)
  S_s_norm  : LU source windows, per-unit representation       (== raw for source:
              the source scaler already maps power to [0,1] by its own capacity,
              so the per-unit ratio for the source is 1.0)
  S_t_norm  : UK target windows, per-unit (target/base_tgt)    (normalized)

Signed log-differences (source minus target), matching the paper's model
  D_raw  = log S_s_raw  - log S_t_raw   =  A + R
  D_norm = log S_s_norm - log S_t_norm  =  R
  A      = D_raw - D_norm               =  nuisance scale discrepancy
  R      = D_norm                       =  residual structural discrepancy

Symmetric mismatch  M = |D|.  The theorem's sufficient condition for
  M_norm <= M_raw   is   A * R >= 0   (same sign / one zero).

Prints A, R, their product, both mismatches, and the observed reduction ratio
exp(M_raw - M_norm)  (the empirical analogue of the 211 -> 2 collapse).
Writes Data/results/sign_condition.csv.
"""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days
from coldstart_transfer.features import SEQ_FEATURES, NEXT_EXO
from coldstart_transfer.ewc import estimate_fisher

scalers = joblib.load("Data/models/ev_scalers.joblib")
PROD = "Data/models/ev_cnn_lstm_20260718.keras"
base_src = float(scalers["scaler_y"].data_max_[0])

tgt = pd.read_csv("Data/uk_ev_features_full.csv")
src = pd.read_csv("Data/lux_source_features.csv")
base_tgt = float(np.quantile(tgt.ev_kw, 0.999))
ratio = base_src / base_tgt
print(f"[sign] base_src={base_src:.1f} base_tgt={base_tgt:.1f} ratio={ratio:.2f}")

# power-carrying feature indices (rescaled by the per-unit ratio)
POWER = ("ev_kw", "lag_1", "lag_4", "lag_96", "lag_672",
         "roll_1h_mean", "roll_6h_mean", "roll_24h_mean")
pw_seq = [i for i, c in enumerate(SEQ_FEATURES) if c in POWER]
pw_nxt = [i for i, c in enumerate(NEXT_EXO) if c in POWER]


def rescale(Xs, Xn, r):
    Xs2, Xn2 = Xs.copy(), Xn.copy()
    Xs2[:, :, pw_seq] *= r
    Xn2[:, pw_nxt] *= r
    return Xs2, Xn2


def s_summary(df, r, n_days=30, seed=0):
    """Mean diagonal Fisher importance on the first n_days of `df`, power features
    scaled by ratio r (r=1 -> raw source-scaler pipeline)."""
    Xs, Xn, y, ts = make_windows(df, scalers)
    tr, _ = split_holdout(Xs, Xn, y, ts, 14)
    xs, xn, yy, _ = first_n_days(*tr, n_days=n_days)
    if r != 1.0:
        xs, xn = rescale(xs, xn, r)
    m = build_ev_model(seed=seed)
    load_production_weights(m, PROD)                 # deployed weights, no retraining
    F = estimate_fisher(m, xs, xn, m_points=500, batch_size=32, seed=seed)
    total = float(sum(float(tf.reduce_sum(f)) for f in F))
    p = int(sum(int(np.prod(f.shape)) for f in F))
    return total / p                                  # mean over ALL diagonal entries


rows = []
for seed in range(3):
    S_s_raw = s_summary(src, 1.0, seed=seed)          # source: per-unit ratio == 1
    S_t_raw = s_summary(tgt, 1.0, seed=seed)          # target through source scaler
    S_s_norm = S_s_raw                                # source unchanged under per-unit
    S_t_norm = s_summary(tgt, ratio, seed=seed)       # target rescaled to per-unit
    D_raw = np.log(S_s_raw) - np.log(S_t_raw)
    D_norm = np.log(S_s_norm) - np.log(S_t_norm)
    A = D_raw - D_norm
    R = D_norm
    rows.append(dict(seed=seed, S_s_raw=S_s_raw, S_t_raw=S_t_raw, S_t_norm=S_t_norm,
                     A=A, R=R, AR=A * R, M_raw=abs(D_raw), M_norm=abs(D_norm)))
    print(f"[seed {seed}] A={A:+.3f} R={R:+.3f} A*R={A*R:+.3f} "
          f"M_raw={abs(D_raw):.3f} M_norm={abs(D_norm):.3f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv("Data/results/sign_condition.csv", index=False)
A_m, A_s = R.A.mean(), R.A.std()
R_m, R_s = R.R.mean(), R.R.std()
Mr, Mn = R.M_raw.mean(), R.M_norm.mean()
print("\n=== SUMMARY (3 seeds) ===")
print(f"A (nuisance scale)     = {A_m:+.3f} +/- {A_s:.3f}")
print(f"R (structural residual)= {R_m:+.3f} +/- {R_s:.3f}")
print(f"sign condition  A*R>=0 : {'HOLDS' if (R.AR > 0).all() else 'FAILS'} "
      f"(all seeds same sign: A>0={all(R.A>0)}, R>0={all(R.R>0)})")
print(f"M_raw = {Mr:.3f} (ratio {np.exp(Mr):.1f}x)   "
      f"M_norm = {Mn:.3f} (ratio {np.exp(Mn):.1f}x)")
print(f"observed collapse exp(M_raw-M_norm) = {np.exp(Mr-Mn):.1f}x")
print("DONE")
