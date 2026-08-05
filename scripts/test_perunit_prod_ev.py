"""
Per-unit WITH THE DEPLOYED PRODUCTION MODEL (reviewer blocker #1).

No source retraining. The frozen production scaler already scales power by the
SOURCE capacity (min-max, min=0 => x/base_src). Per-unit just rescales the target
by its OWN capacity: multiply the power features (and target) by
ratio = base_src/base_tgt AFTER the production scaling, so the deployed model sees
the target occupying the same [0,1] range the source did -- instead of a tiny
corner. Weather/time keep the production scaling (the model expects them).

Compares, both warm-started from the production weights at N=30 (3 seeds):
  B3_minmax   : current pipeline (target through source scaler)  -> ~0.525 nRMSE
  B3_perunit  : same model, target rescaled to per-unit-target
"""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.model import build_ev_model, load_production_weights
from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days, inverse_target
from coldstart_transfer.features import SEQ_FEATURES, NEXT_EXO

scalers = joblib.load("Data/models/ev_scalers.joblib")
PROD = "Data/models/ev_cnn_lstm_20260718.keras"
base_src = float(scalers["scaler_y"].data_max_[0])            # 578.4
tgt = pd.read_csv("Data/uk_ev_features_full.csv")
base_tgt = float(np.quantile(tgt.ev_kw, 0.999))
ratio = base_src / base_tgt
print(f"[perunit-prod] base_src={base_src:.1f} base_tgt={base_tgt:.1f} ratio={ratio:.2f}")

Xs, Xn, y, ts = make_windows(tgt, scalers)                    # source-scaled (current pipeline)
pw_seq = [i for i, c in enumerate(SEQ_FEATURES) if c in ("ev_kw","lag_1","lag_4","lag_96","lag_672","roll_1h_mean","roll_6h_mean","roll_24h_mean")]
pw_nxt = [i for i, c in enumerate(NEXT_EXO) if c in ("lag_1","lag_4","lag_96","lag_672","roll_1h_mean","roll_6h_mean","roll_24h_mean")]

def rescale(Xs, Xn, y):
    Xs2 = Xs.copy(); Xn2 = Xn.copy()
    Xs2[:, :, pw_seq] *= ratio; Xn2[:, pw_nxt] *= ratio
    return Xs2, Xn2, y * ratio

def evaluate(model, Xh, Nh, yh, per_unit):
    p = np.clip(model.predict([Xh, Nh], verbose=0).reshape(-1), 0, None)
    if per_unit:
        pred_kw = p / ratio * base_src; true_kw = yh / ratio * base_src   # undo ratio + source scale
    else:
        pred_kw = inverse_target(p, scalers); true_kw = inverse_target(yh, scalers)
    rmse = float(np.sqrt(np.mean((pred_kw - true_kw) ** 2))); mean = float(np.mean(true_kw))
    return rmse / mean, rmse

for mode in ("minmax", "perunit"):
    A = rescale(Xs, Xn, y) if mode == "perunit" else (Xs, Xn, y)
    tr, ho = split_holdout(A[0], A[1], A[2], ts, 14)
    xs, xn, yy, _ = first_n_days(*tr, n_days=30)
    for seed in range(3):
        keras.utils.set_random_seed(seed)
        m = build_ev_model(seed=seed); load_production_weights(m, PROD)   # WARM from deployed model
        m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
        m.fit([xs, xn], yy, epochs=10, batch_size=64, verbose=0)
        nr, rk = evaluate(m, ho[0], ho[1], ho[2], per_unit=(mode == "perunit"))
        print(f"[B3_{mode} s={seed}] nRMSE={nr:.4f} RMSE={rk:.2f}kW", flush=True)
print("DONE")
