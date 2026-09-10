"""
ev_demand_forecaster.py -- day-ahead EV demand from the Service 1 production model.

The MPC needs tomorrow's charging demand. The deployed CNN-LSTM is a ONE-STEP
model (given 672 past steps, predict the next one), so a 96-step day-ahead
profile requires rolling the model forward on its own output, recomputing the
autoregressive features (lags and rolling means) at every step.

DATA: Data/lux_source_features.csv is the SAME site as
Data/ECC_master_PV_EMOB1_EMOB2_15min.csv -- verified identical on the 47,524
overlapping rows (corr 1.0, max abs diff 0.0) -- so the production features
(including weather) are already available and no re-engineering is needed.

TARGET LEAKAGE, and how it is handled here
------------------------------------------
features.engineer_features defines the rolling means as TRAILING AND INCLUSIVE:
    roll_1h_mean[t] = mean(ev[t-3..t])
and NEXT_EXO passes the predicted step's features to the model. The model is
therefore trained with (ev[t-3]+ev[t-2]+ev[t-1]+ev[t])/4 available as an input
while predicting ev[t]: a partial leak of the target (the same holds for the 6 h
and 24 h means). At true forecast time ev[t] is unknown, so the leak cannot be
reproduced. We substitute the last known/predicted value for ev[t] inside the
rolling windows -- the honest choice, and the one an operational deployment must
make. Expect the recursive forecast to be worse than the one-step scores in the
paper, precisely because those scores benefit from the leak.

Writes nothing; import `forecast_day` or run for a self-test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from coldstart_transfer.features import SEQ_FEATURES, NEXT_EXO
from coldstart_transfer.windowing import LOOKBACK

EVKW = SEQ_FEATURES.index("ev_kw")
IDX = {c: SEQ_FEATURES.index(c) for c in SEQ_FEATURES}
LAGS = [(1, "lag_1"), (4, "lag_4"), (96, "lag_96"), (672, "lag_672")]
ROLLS = [(4, "roll_1h_mean"), (24, "roll_6h_mean"), (96, "roll_24h_mean")]


def _refresh_autoregressive(row, hist, ev_now):
    """Recompute lag/roll entries of one feature row from the ev history.

    hist   : 1-D array of ev values up to (but excluding) the step being built
    ev_now : value standing in for ev[t] inside the inclusive rolling windows
             (unknown at forecast time -> last known/predicted value)
    """
    for k, col in LAGS:
        row[IDX[col]] = hist[-k] if len(hist) >= k else 0.0
    for w, col in ROLLS:
        tail = hist[-(w - 1):] if w > 1 and len(hist) >= w - 1 else hist
        vals = np.concatenate([tail, [ev_now]]) if len(tail) else np.array([ev_now])
        row[IDX[col]] = float(vals.mean())
    return row


def forecast_day(model, scalers, feats_raw, start_idx, horizon=96):
    """Roll the one-step model forward `horizon` steps from `start_idx`.

    feats_raw : (N, 21) unscaled SEQ_FEATURES array (exogenous columns are real
                future values -- weather and calendar are known day-ahead)
    start_idx : index of the FIRST step to predict; needs >= LOOKBACK history
    returns   : (horizon,) predicted ev_kw
    """
    if start_idx < LOOKBACK:
        raise ValueError("need LOOKBACK history before start_idx")

    sc_seq, sc_next, sc_y = (scalers["scaler_seq"], scalers["scaler_next"],
                             scalers["scaler_y"])
    work = feats_raw.copy()
    hist = list(work[:start_idx, EVKW].astype(float))
    preds = []

    for h in range(horizon):
        t = start_idx + h
        ev_now = hist[-1] if hist else 0.0
        work[t] = _refresh_autoregressive(work[t].copy(), np.asarray(hist), ev_now)

        seq = work[t - LOOKBACK:t]                       # 672 past steps
        x_seq = sc_seq.transform(seq).astype(np.float32)[None, ...]
        nxt = work[t][1:]                                # NEXT_EXO = drop ev_kw
        x_next = sc_next.transform(nxt[None, :]).astype(np.float32)

        y_s = float(model([x_seq, x_next], training=False).numpy().ravel()[0])
        y_kw = float(sc_y.inverse_transform([[y_s]])[0, 0])
        y_kw = max(0.0, y_kw)

        preds.append(y_kw)
        work[t, EVKW] = y_kw
        hist.append(y_kw)

    return np.asarray(preds)


def _self_test():
    import joblib
    from coldstart_transfer.model import build_ev_model, load_production_weights

    df = pd.read_csv("Data/lux_source_features.csv")
    feats = df[SEQ_FEATURES].to_numpy(np.float32)
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")

    n_days = (len(feats) - LOOKBACK) // 96
    print(f"[evfc] {len(feats)} rows, {n_days} forecastable days")
    for d in range(n_days - 3, n_days):
        s = LOOKBACK + d * 96
        if s + 96 > len(feats):
            break
        pred = forecast_day(m, scalers, feats, s)
        real = feats[s:s + 96, EVKW].astype(float)
        pe, re_ = pred.sum() * 0.25, real.sum() * 0.25
        print(f"  day {d}: predicted {pe:7.1f} kWh | real {re_:7.1f} kWh | "
              f"err {100*(pe-re_)/max(re_,1e-9):+6.1f}%")


if __name__ == "__main__":
    _self_test()
