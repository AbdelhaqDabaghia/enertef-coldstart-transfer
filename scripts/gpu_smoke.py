import time, joblib, pandas as pd, tensorflow as tf
print("GPU visible:", tf.config.list_physical_devices('GPU'))
from coldstart_transfer.model import fidelity_check
from coldstart_transfer.windowing import make_windows, split_holdout
from coldstart_transfer.trainer import run_condition

fidelity_check("Data/models/ev_cnn_lstm_20260718.keras")

scalers = joblib.load("Data/models/ev_scalers.joblib")
df = pd.read_csv("Data/uk_ev_features_full.csv")
Xs, Xn, y, ts = make_windows(df, scalers)
tr, ho = split_holdout(Xs, Xn, y, ts, holdout_days=14)

t0 = time.time()
r = run_condition("B3", tr, ho, scalers, "Data/models/ev_cnn_lstm_20260718.keras",
                  n_days=7, epochs=2, batch_size=64, seed=0)
r.pop("model")
print(f"[GPU] B3 N=7 2ep: nRMSE={r['target_nrmse']:.4f}  wall={time.time()-t0:.1f}s "
      f"(CPU baseline was ~45.5s)")
