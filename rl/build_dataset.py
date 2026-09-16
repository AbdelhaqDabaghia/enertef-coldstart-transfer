"""
build_dataset.py -- assemble every complete day the site actually has.

Writes rl/data/site_days.npz with one row per COMPLETE day (96 quarter-hours,
no gaps, both charging sites and PV present):

    ev1, ev2   (D, 96)  realised consumption per site, kW
    pv         (D, 96)  realised PV production, kW
    f1, f2     (D, 96)  the FORECAST the controller would have had, split by
                        capacity share exactly as the deployed MPC splits it
    dates      (D,)     ISO date, so nothing is ever aligned by position

and rl/data/prices.npz with

    price      (P, 96)  EUR/MWh, one row per complete price day
    dates      (P,)

Demand and prices are kept in SEPARATE files on purpose. There are far more
price days than demand days, and the environment pairs them by sampling, which
turns 494 demand days into hundreds of thousands of training scenarios and
forces the policy to answer the price SHAPE rather than memorise a date. The
evaluation split pairs them deterministically instead, so the reported number
is reproducible.

Honest note on size: the site has 3.05 years of file coverage but only 494 days
where PV, EMOB1 and EMOB2 are all complete -- EMOB2 is populated 44 % of the
time. There is no four-year demand history. The four years are on the price
side, where they matter most for arbitrage.

    python -m rl.build_dataset
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

SITE = os.environ.get("SITE", "Data/ECC_master_PV_EMOB1_EMOB2_15min.csv")
FEATURES = os.environ.get("FEATURES", "Data/lux_source_features_causal.csv")
MODEL = os.environ.get("MODEL", "Data/models/ev_cnn_lstm_causal_full_warm_e20_s1.keras")
SCALERS = os.environ.get("SCALERS", "Data/models/ev_scalers_causal_full_warm_e20_s1.joblib")
PRICES = os.environ.get("PRICES", "Data/entsoe_dayahead_DE_LU_4y.csv")
PRICES_FALLBACK = r"C:\dev\MPC_RL\BESS_MPC_PROJECT\prices_da_real.csv"
OUTDIR = os.environ.get("OUTDIR", "rl/data")

H = 96
CAP1 = 4 * 300.0
CAP2 = 4 * 400.0 + 8 * 22.0
SHARE1 = CAP1 / (CAP1 + CAP2)
SHARE2 = 1.0 - SHARE1


def complete_days(df, cols):
    """Days with all 96 steps present and no NaN in `cols`."""
    d = df.copy()
    d["day"] = d.index.date
    ok = d[cols].notna().all(axis=1)
    counts = ok.groupby(d["day"]).sum()
    return sorted(counts[counts >= H].index)


def build_site():
    df = pd.read_csv(SITE, parse_dates=["Started at"])
    df["t"] = pd.to_datetime(df["Started at"], utc=True)
    df = df.set_index("t").sort_index()
    cols = ["PV_TotalProduction_kW", "EMOB1_EV_Consumption_kW",
            "EMOB2_EV_Consumption_kW"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    days = complete_days(df, cols)
    print("[data] %d complete days (PV + EMOB1 + EMOB2) = %.2f years"
          % (len(days), len(days) / 365.0), flush=True)

    ev1, ev2, pv, keep = [], [], [], []
    for day in days:
        sl = df[df.index.date == day]
        if len(sl) < H:
            continue
        sl = sl.iloc[:H]
        ev1.append(sl["EMOB1_EV_Consumption_kW"].to_numpy(float))
        ev2.append(sl["EMOB2_EV_Consumption_kW"].to_numpy(float))
        pv.append(sl["PV_TotalProduction_kW"].to_numpy(float))
        keep.append(str(day))
    return np.array(ev1), np.array(ev2), np.array(pv), np.array(keep)


def build_forecast(dates):
    """The forecast the deployed controller would have had, per day.

    Runs the promoted causal EV model over the feature frame and indexes the
    result by date, so a day that the model cannot cover (not enough lookback)
    is dropped rather than silently filled.
    """
    from coldstart_transfer.windowing import make_windows, inverse_target
    from coldstart_transfer.model import build_ev_model

    scalers = joblib.load(SCALERS)
    sdf = pd.read_csv(FEATURES)
    Xs, Xn, _, ts = make_windows(sdf, scalers)

    m = build_ev_model()
    m.load_weights(MODEL)
    pred = m.predict([Xs, Xn], verbose=0, batch_size=512).reshape(-1)
    ev_fc = np.clip(inverse_target(np.clip(pred, 0, None), scalers), 0, None)

    idx = pd.to_datetime(ts, utc=True)
    s = pd.Series(ev_fc, index=idx)
    by_day = {}
    for day, grp in s.groupby(s.index.date):
        if len(grp) >= H:
            by_day[str(day)] = grp.to_numpy()[:H]

    have = np.array([d in by_day for d in dates])
    fc = np.zeros((len(dates), H))
    for i, d in enumerate(dates):
        if have[i]:
            fc[i] = by_day[d]
    print("[data] forecast available for %d / %d days" % (have.sum(), len(dates)),
          flush=True)
    return fc, have


def build_prices():
    path = PRICES if os.path.exists(PRICES) else PRICES_FALLBACK
    if path == PRICES_FALLBACK:
        print("[data] WARNING: %s absent, falling back to the 15-day file.\n"
              "       Run `python -m rl.fetch_prices_4y` for the real four years."
              % PRICES, flush=True)
    p = pd.read_csv(path)
    col = "price" if "price" in p.columns else p.columns[-1]
    p["t"] = pd.to_datetime(p["timestamp"], utc=True)
    p = p.set_index("t").sort_index()
    p[col] = pd.to_numeric(p[col], errors="coerce")
    if p[col].mean() < 5:                      # EUR/kWh -> EUR/MWh
        p[col] = p[col] * 1000.0

    rows, dates = [], []
    for day, grp in p.groupby(p.index.date):
        v = grp[col].to_numpy(float)
        if len(v) >= H and np.isfinite(v[:H]).all():
            rows.append(v[:H])
            dates.append(str(day))
    print("[data] %d complete price days = %.2f years"
          % (len(rows), len(rows) / 365.0), flush=True)
    return np.array(rows), np.array(dates)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    ev1, ev2, pv, dates = build_site()
    fc, have = build_forecast(dates)
    ev1, ev2, pv, dates, fc = ev1[have], ev2[have], pv[have], dates[have], fc[have]

    # Splitting the total EV forecast between the two metering points.
    #
    # Production splits it by CHARGER CAPACITY: SHARE1 = 0.403, SHARE2 = 0.597.
    # Observed consumption splits the other way -- EMOB1 takes about 63 % and
    # EMOB2 about 37 %. Planning a reduction on EMOB2 that its real demand
    # cannot absorb is truncated by the physical floor at zero, while the
    # matching increase elsewhere applies in full, so the mismatch loses money
    # on every day it occurs. That is a genuine defect in the deployed system
    # and it belongs in the deployment audit.
    #
    # Here it is a confound: it would make every MPC arm look bad for a reason
    # that has nothing to do with the controller being compared. The split is
    # therefore estimated from the TRAINING days only -- never the held-out
    # ones -- so the comparison measures control, not a mis-set constant.
    n_eval = int(os.environ.get("EVAL_DAYS_HELD", "120"))
    tr = slice(0, max(len(dates) - n_eval, 1))
    emp1 = float(ev1[tr].sum() / max((ev1[tr].sum() + ev2[tr].sum()), 1e-9))
    print("[data] site split -- production uses capacity shares %.3f / %.3f, "
          "training days observe %.3f / %.3f"
          % (SHARE1, SHARE2, emp1, 1 - emp1), flush=True)

    np.savez_compressed(
        os.path.join(OUTDIR, "site_days.npz"),
        ev1=ev1, ev2=ev2, pv=pv, dates=dates,
        f1=emp1 * fc, f2=(1.0 - emp1) * fc,
        share_emp=np.array([emp1, 1.0 - emp1]),
        share_capacity=np.array([SHARE1, SHARE2]),
    )
    price, pdates = build_prices()
    np.savez_compressed(os.path.join(OUTDIR, "prices.npz"),
                        price=price, dates=pdates)

    err = (fc - (ev1 + ev2))
    print("\n[data] wrote %s : %d days" % (OUTDIR, len(dates)))
    print("[data] demand %s -> %s" % (dates[0], dates[-1]))
    print("[data] EV realised mean %.1f kW | forecast mean %.1f kW | bias %+.1f kW"
          % ((ev1 + ev2).mean(), fc.mean(), err.mean()))
    print("[data] PV mean %.1f kW" % pv.mean())
    print("[data] prices %s -> %s" % (pdates[0], pdates[-1]))
    print("[data] scenario space: %d demand days x %d price days = %d pairs"
          % (len(dates), len(pdates), len(dates) * len(pdates)))


if __name__ == "__main__":
    main()
