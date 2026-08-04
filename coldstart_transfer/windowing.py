"""
windowing.py -- turn a flat feature frame (uk_ev_features_full.csv or a
reconstructed Luxembourg source frame) into scaled training triples:

    X_seq : (N, 672, 21)  scaled SEQ_FEATURES lookback
    X_next: (N, 20)       scaled NEXT_EXO features at the predicted step
    y     : (N,)          scaled ev_kw target at the predicted step

Convention replicates production _assemble_pv/ev_arrays exactly:
    for i in range(len(df) - LOOKBACK):
        X_seq[i]  = feats[i : i+LOOKBACK]
        X_next[i] = next_feats[i+LOOKBACK]
        y[i]      = target[i+LOOKBACK]

Scaling uses the FROZEN production scalers (scaler_seq/scaler_next/scaler_y) for
ALL conditions -- never re-fit (that would break weight compatibility with the
transferred model). See [[enertef-trainer-image-gotchas]].
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .features import SEQ_FEATURES, NEXT_EXO  # canonical feature order

LOOKBACK = 672


def scale_frame(df: pd.DataFrame, scalers: dict):
    """Return (seq_scaled[N,21], next_scaled[N,20], y_scaled[N]) column-scaled by
    the frozen production scalers. df must contain SEQ_FEATURES (incl. ev_kw)."""
    seq_raw = df[SEQ_FEATURES].to_numpy(dtype=np.float32)
    next_raw = df[NEXT_EXO].to_numpy(dtype=np.float32)
    y_raw = df[["ev_kw"]].to_numpy(dtype=np.float32)
    seq_s = scalers["scaler_seq"].transform(seq_raw).astype(np.float32)
    next_s = scalers["scaler_next"].transform(next_raw).astype(np.float32)
    y_s = scalers["scaler_y"].transform(y_raw).astype(np.float32).reshape(-1)
    return seq_s, next_s, y_s


def make_windows(df: pd.DataFrame, scalers: dict):
    """Flat frame -> (X_seq, X_next, y) scaled triples, production convention."""
    seq_s, next_s, y_s = scale_frame(df, scalers)
    n = len(df) - LOOKBACK
    if n <= 0:
        raise ValueError(f"need > {LOOKBACK} rows, got {len(df)}")
    X_seq = np.zeros((n, LOOKBACK, len(SEQ_FEATURES)), dtype=np.float32)
    X_next = np.zeros((n, len(NEXT_EXO)), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        X_seq[i] = seq_s[i:i + LOOKBACK]
        X_next[i] = next_s[i + LOOKBACK]
        y[i] = y_s[i + LOOKBACK]
    # timestamp of each predicted step (for holdout splitting / logging)
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()[LOOKBACK:]
    return X_seq, X_next, y, ts


def inverse_target(y_scaled: np.ndarray, scalers: dict) -> np.ndarray:
    """Un-scale a target/prediction back to kW."""
    y = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
    return scalers["scaler_y"].inverse_transform(y).reshape(-1)


def split_holdout(X_seq, X_next, y, ts, holdout_days: int = 14):
    """Fixed FINAL holdout (most recent `holdout_days`), excluded from training.
    Returns (train_tuple, holdout_tuple)."""
    steps = holdout_days * 96  # 15-min resolution
    if steps >= len(y):
        raise ValueError(f"holdout {steps} >= dataset {len(y)}")
    tr = (X_seq[:-steps], X_next[:-steps], y[:-steps], ts[:-steps])
    ho = (X_seq[-steps:], X_next[-steps:], y[-steps:], ts[-steps:])
    return tr, ho


def first_n_days(X_seq, X_next, y, ts, n_days: int):
    """Take the FIRST n_days of a (already holdout-excluded) training set, to
    simulate the accumulating daily stream of RQ1 cold-start checkpoints."""
    steps = n_days * 96
    steps = min(steps, len(y))
    return X_seq[:steps], X_next[:steps], y[:steps], ts[:steps]


if __name__ == "__main__":
    import joblib
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    df = pd.read_csv("Data/uk_ev_features_full.csv")
    Xs, Xn, y, ts = make_windows(df, scalers)
    print(f"[windowing] UK: {len(df)} rows -> {len(y)} windows | "
          f"X_seq{Xs.shape} X_next{Xn.shape} y{y.shape}")
    (tr, ho) = split_holdout(Xs, Xn, y, ts, holdout_days=14)
    print(f"[windowing] train={len(tr[2])}  holdout(14d)={len(ho[2])}")
    print(f"[windowing] scaled y range [{y.min():.3f},{y.max():.3f}] "
          f"| kW range [{inverse_target(y, scalers).min():.2f},"
          f"{inverse_target(y, scalers).max():.2f}]")
