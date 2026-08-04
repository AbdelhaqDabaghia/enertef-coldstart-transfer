"""
pv.py -- PV replication of the cold-start protocol. Mirrors the EV modules but
for the deployed PV LSTM (pv_lstm_openmeteo.keras): LOOKBACK=96 (1 day), 11
sequential features, 10 next-step exogenous features.

PV_SEQ_FEATURES (order from pv_metadata.json):
  pv_kw, shortwave/direct/diffuse radiation, cloud_cover, temperature_2m,
  wind_speed_10m, tod_sin, tod_cos, doy_sin, doy_cos
(No lags/rolls and no day-of-week: PV is weather/astronomy driven.)

Architecture (verified against the saved model, 148,161 params):
  seq(96,11) -> LSTM(128, return_seq) -> BN -> LSTM(64) -> BN -> Dropout(0.3)
  next(10)   -> Dense(64) -> BN -> Dropout(0.2)
  concat -> Dense(128) -> BN -> Dropout(0.3) -> Dense(64) -> Dense(1, linear)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PV_LOOKBACK = 96
PV_SEQ_FEATURES = [
    "pv_kw", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloud_cover", "temperature_2m", "wind_speed_10m",
    "tod_sin", "tod_cos", "doy_sin", "doy_cos",
]
PV_NEXT_EXO = PV_SEQ_FEATURES[1:]   # 10 features (drop pv_kw)
PV_WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
              "cloud_cover", "temperature_2m", "wind_speed_10m"]


def build_pv_model(seed: int | None = None) -> keras.Model:
    if seed is not None:
        keras.utils.set_random_seed(seed)
    seq_in = keras.Input(shape=(PV_LOOKBACK, len(PV_SEQ_FEATURES)), name="seq_input")
    x = layers.LSTM(128, activation="tanh", recurrent_activation="sigmoid",
                    return_sequences=True)(seq_in)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    next_in = keras.Input(shape=(len(PV_NEXT_EXO),), name="next_input")
    y = layers.Dense(64, activation="relu")(next_in)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation="relu")(z)
    out = layers.Dense(1, activation="linear", name="pv_next")(z)
    return keras.Model([seq_in, next_in], out, name="pv_lstm")


def load_pv_production_weights(model, keras_path):
    prod = keras.models.load_model(keras_path, compile=False)
    model.set_weights(prod.get_weights())
    return model


def fidelity_check_pv(keras_path, tol=1e-4):
    prod = keras.models.load_model(keras_path, compile=False)
    m = build_pv_model()
    if len(m.get_weights()) != len(prod.get_weights()):
        raise AssertionError(f"weight count {len(m.get_weights())} != {len(prod.get_weights())}")
    for i, (a, b) in enumerate(zip(m.get_weights(), prod.get_weights())):
        if a.shape != b.shape:
            raise AssertionError(f"weight[{i}] {a.shape} != {b.shape}")
    m.set_weights(prod.get_weights())
    rng = np.random.default_rng(0)
    s = rng.standard_normal((8, PV_LOOKBACK, len(PV_SEQ_FEATURES))).astype("float32")
    n = rng.standard_normal((8, len(PV_NEXT_EXO))).astype("float32")
    err = float(np.max(np.abs(prod.predict([s, n], verbose=0) - m.predict([s, n], verbose=0))))
    if err > tol:
        raise AssertionError(f"prediction mismatch max|d|={err:.2e} > {tol}")
    print(f"[pv] fidelity OK: build_pv_model reproduces {keras_path} "
          f"(max|d|={err:.2e}, {len(m.get_weights())} weight tensors).")


def engineer_pv_features(df, timestamp_col="timestamp"):
    """Build the 11 PV features from raw (timestamp, pv_kw, 6 weather)."""
    d = df.sort_values(timestamp_col).reset_index(drop=True)
    ts = pd.to_datetime(d[timestamp_col], utc=True)
    out = pd.DataFrame({timestamp_col: d[timestamp_col].values, "pv_kw": d["pv_kw"].astype(float).values})
    for c in PV_WEATHER:
        out[c] = d[c].astype(float).values
    minutes = ts.dt.hour * 60 + ts.dt.minute
    doy = ts.dt.dayofyear.astype(float)
    out["tod_sin"] = np.sin(2 * np.pi * minutes / 1440).values
    out["tod_cos"] = np.cos(2 * np.pi * minutes / 1440).values
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).values
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).values
    return out[[timestamp_col] + PV_SEQ_FEATURES]


def make_pv_windows(df, scalers):
    """Flat PV frame -> (X_seq[N,96,11], X_next[N,10], y[N], ts) scaled triples."""
    seq = scalers["scaler_seq"].transform(df[PV_SEQ_FEATURES].to_numpy(np.float32)).astype(np.float32)
    nxt = scalers["scaler_next"].transform(df[PV_NEXT_EXO].to_numpy(np.float32)).astype(np.float32)
    ys = scalers["scaler_y"].transform(df[["pv_kw"]].to_numpy(np.float32)).astype(np.float32).reshape(-1)
    n = len(df) - PV_LOOKBACK
    X_seq = np.zeros((n, PV_LOOKBACK, len(PV_SEQ_FEATURES)), np.float32)
    X_next = np.zeros((n, len(PV_NEXT_EXO)), np.float32)
    y = np.zeros((n,), np.float32)
    for i in range(n):
        X_seq[i] = seq[i:i + PV_LOOKBACK]
        X_next[i] = nxt[i + PV_LOOKBACK]
        y[i] = ys[i + PV_LOOKBACK]
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()[PV_LOOKBACK:]
    return X_seq, X_next, y, ts


if __name__ == "__main__":
    import sys
    fidelity_check_pv(sys.argv[1] if len(sys.argv) > 1 else "Data/models/pv_lstm_openmeteo.keras")
