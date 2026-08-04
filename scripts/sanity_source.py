"""Pre-flight sanity check: reload the 2026-07-18 production model + frozen
scalers and confirm a NON-DEGENERATE, reasonable MAE on the Luxembourg source
holdout, guarding against scaler/architecture mismatch before transfer runs.
Reference (prod gate, 7-day holdout): promoted EWC MAE 13.58 kW / baseline 42.04."""
import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from coldstart_transfer.windowing import make_windows, split_holdout, inverse_target

print("GPU:", tf.config.list_physical_devices('GPU'))
scalers = joblib.load("Data/models/ev_scalers.joblib")
df = pd.read_csv("Data/lux_source_features.csv")
Xs, Xn, y, ts = make_windows(df, scalers)
model = keras.models.load_model("Data/models/ev_cnn_lstm_20260718.keras", compile=False)

def mae_kw(Xs, Xn, y):
    p = np.clip(model.predict([Xs, Xn], verbose=0).reshape(-1), 0, None)
    return float(np.mean(np.abs(inverse_target(p, scalers) - inverse_target(y, scalers))))

# last 7 and 90 days of the source window
for days in (7, 90):
    steps = days*96
    m = mae_kw(Xs[-steps:], Xn[-steps:], y[-steps:])
    print(f"[sanity] source holdout last {days}d: MAE={m:.3f} kW  (n={steps})")
print("[sanity] reference (prod 7d gate): EWC 13.58 / baseline 42.04 kW")
