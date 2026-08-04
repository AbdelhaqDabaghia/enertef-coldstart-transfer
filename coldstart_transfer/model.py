"""
model.py -- EV CNN-LSTM builder, EXACT replica of the Luxembourg production
architecture (ev_cnn_lstm_20260718.keras, the 2026-07-18 EWC-promoted model).

Verified layer-for-layer against the saved model:
    seq_input(672,21) -> Conv1D(64,k=5,causal,relu) -> Conv1D(64,k=5,causal,relu)
                      -> LSTM(64, tanh, rec=sigmoid) -> Dropout(0.3)
    next_input(20)    -> Dense(64, relu) -> Dropout(0.2)
    concat([lstm_branch, next_branch]) -> Dense(128, relu) -> Dropout(0.3)
                                       -> Dense(1, LINEAR)

Output is LINEAR (matches production); non-negativity is enforced POST-HOC via
predict_nonneg() (clip at 0), per the task spec. Loss is Huber(delta=0.5).

Because the graph is built in the same topological order as the saved model,
`new_model.set_weights(prod_model.get_weights())` transfers production weights
1:1 -- this is the warm-start used by B1/B3/B4 and validated by fidelity_check().
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LOOKBACK = 672
N_SEQ_FEATURES = 21
N_NEXT_FEATURES = 20
HUBER_DELTA = 0.5


def build_ev_model(seed: int | None = None) -> keras.Model:
    """Build the production EV CNN-LSTM. If seed is given, weight init is
    reproducible (used by B0 from-scratch runs)."""
    if seed is not None:
        keras.utils.set_random_seed(seed)

    seq_in = keras.Input(shape=(LOOKBACK, N_SEQ_FEATURES), name="seq_input")
    x = layers.Conv1D(64, 5, padding="causal", activation="relu")(seq_in)
    x = layers.Conv1D(64, 5, padding="causal", activation="relu")(x)
    x = layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid")(x)
    x = layers.Dropout(0.3)(x)

    next_in = keras.Input(shape=(N_NEXT_FEATURES,), name="next_input")
    y = layers.Dense(64, activation="relu")(next_in)
    y = layers.Dropout(0.2)(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation="linear", name="ev_next")(z)

    return keras.Model(inputs=[seq_in, next_in], outputs=out, name="ev_cnn_lstm")


def compile_ev_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    """Compile with the production optimizer/loss (Adam 1e-3, Huber delta=0.5).
    NOTE: EWC/replay runs use a custom train step, not this compiled loss."""
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.Huber(delta=HUBER_DELTA))
    return model


def load_production_weights(model: keras.Model, keras_path: str) -> keras.Model:
    """Warm-start: copy the saved production model's weights into `model`.
    Relies on identical topology (same shapes/order). Raises if shapes differ."""
    prod = keras.models.load_model(keras_path, compile=False)
    model.set_weights(prod.get_weights())
    return model


def predict_nonneg(model: keras.Model, inputs) -> np.ndarray:
    """Production-equivalent inference: linear head then clip at 0 (non-negative
    EV power). Returns a 1-D array of predictions in scaled space."""
    y = model.predict(inputs, verbose=0).reshape(-1)
    return np.clip(y, 0.0, None)


def fidelity_check(keras_path: str, tol: float = 1e-5) -> None:
    """HARD check: build_ev_model + set_weights(production) must reproduce the
    saved model's outputs on random input. Guarantees warm-start validity before
    any transfer experiment. Raises AssertionError on mismatch."""
    prod = keras.models.load_model(keras_path, compile=False)
    m = build_ev_model()
    if len(m.get_weights()) != len(prod.get_weights()):
        raise AssertionError(
            f"weight-tensor count differs: built={len(m.get_weights())} "
            f"prod={len(prod.get_weights())} -> architecture mismatch")
    for i, (a, b) in enumerate(zip(m.get_weights(), prod.get_weights())):
        if a.shape != b.shape:
            raise AssertionError(f"weight[{i}] shape {a.shape} != prod {b.shape}")
    m.set_weights(prod.get_weights())

    rng = np.random.default_rng(0)
    seq = rng.standard_normal((8, LOOKBACK, N_SEQ_FEATURES)).astype("float32")
    nxt = rng.standard_normal((8, N_NEXT_FEATURES)).astype("float32")
    ya = prod.predict([seq, nxt], verbose=0)
    yb = m.predict([seq, nxt], verbose=0)
    err = float(np.max(np.abs(ya - yb)))
    if err > tol:
        raise AssertionError(f"prediction mismatch after set_weights: max|d|={err:.2e} > {tol}")
    print(f"[model] fidelity OK: build_ev_model reproduces {keras_path} "
          f"(max|d|={err:.2e}, {len(m.get_weights())} weight tensors).")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Data/models/ev_cnn_lstm_20260718.keras"
    fidelity_check(path)
    build_ev_model().summary()
