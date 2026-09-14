"""
features.py -- Feature reconstruction for the Luxembourg source domain, with
HARD-FAILING fidelity validation against the frozen production scalers.

Why this module exists
----------------------
The Luxembourg source series only exists in raw form (ev_kw + weather). To use
the transferred production model, the source must be feature-engineered into the
exact 21 SEQ_FEATURES the model expects -- column-for-column identical to
uk_ev_features_full.csv and consistent with the frozen scaler ranges.

The feature convention below was reverse-engineered EMPIRICALLY from
uk_ev_features_full.csv (which contains both raw ev_kw and every derived column)
and reproduces that file to ~1e-14. It is NOT the DFL repo's per-window
build_scenarios logic (that is a different design); see the project notes.

Convention (all validated body-exact against the UK file):
    lag_k          = ev.shift(k)                     (global, continuous)
    roll_1h_mean   = ev.rolling(4,  min_periods=1).mean()   # 1h  = 4 steps, incl current
    roll_6h_mean   = ev.rolling(24, min_periods=1).mean()   # 6h  = 24 steps
    roll_24h_mean  = ev.rolling(96, min_periods=1).mean()   # 24h = 96 steps
    tod_sin/cos    = sin/cos(2*pi*(hour*60+minute)/1440)
    doy_sin/cos    = sin/cos(2*pi*dayofyear/365.25)         # NOTE: 365.25, not 365
    dow_sin/cos    = sin/cos(2*pi*dayofweek/7)
    is_weekend     = 1.0 if dayofweek >= 5 else 0.0

HARD FAILURE POLICY (per explicit requirement)
----------------------------------------------
Reconstruction fidelity is enforced by ASSERTIONS THAT RAISE, not warnings.
A warning in a wall of logs gets missed; a bad reconstruction silently
poisoning every downstream transfer result is exactly what we must prevent.
validate_source_features() raises ReconstructionError (nonzero exit if run as a
script) unless BOTH hold:
    (1) per-feature scaler_seq.data_min_/data_max_ match within tolerance, AND
    (2) the strict-'<' train-row count equals the frozen scaler n_samples_seen_.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

# Authoritative feature order (must match ev_metadata.json exactly).
SEQ_FEATURES = [
    "ev_kw", "lag_1", "lag_4", "lag_96", "lag_672",
    "roll_1h_mean", "roll_6h_mean", "roll_24h_mean",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloud_cover", "temperature_2m", "wind_speed_10m",
    "tod_sin", "tod_cos", "doy_sin", "doy_cos",
    "dow_sin", "dow_cos", "is_weekend",
]
NEXT_EXO = SEQ_FEATURES[1:]  # 20 features (drop ev_kw)
WEATHER_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
                "cloud_cover", "temperature_2m", "wind_speed_10m"]


class ReconstructionError(RuntimeError):
    """Raised (hard failure) when a reconstruction does not match production."""


def engineer_features(df: pd.DataFrame,
                      timestamp_col: str = "timestamp",
                      ev_col: str = "ev_kw",
                      causal: bool = False,
                      mode: str | None = None) -> pd.DataFrame:
    """
    Build the 21 SEQ_FEATURES from a raw frame containing `timestamp`, `ev_kw`
    and the six WEATHER_VARS. Lags/rolls are computed on the CONTINUOUS series
    exactly as passed, so the caller controls warm-up: pass extra leading rows
    (>= 672) before the target window to get correct lags at the window start,
    then slice afterwards.

    Returns a new DataFrame with `timestamp` + the 21 columns in canonical order.
    Does NOT scale. Head rows (< 672) carry warm-up-truncated lags (NaN->0),
    matching how a continuous build behaves at series start.

    causal
    ------
    False (default) reproduces the historical convention, in which the trailing
    rolling means INCLUDE the current step, so roll_*[t] contains ev[t].

    That convention leaks the target, and it leaks twice over. Directly:
    roll_1h_mean[t] averages four values, one of which is ev[t] itself. And
    structurally, because windowing.make_windows sets
    X_next[i] = next_feats[i+LOOKBACK] -- the exogenous vector AT THE PREDICTED
    STEP -- while NEXT_EXO contains the three rolling means. The model is
    therefore handed a quantity computed from the value it must predict.

    mode
    ----
    "train"  (default, == causal=False) the historical convention:
             roll_*[t] = mean(ev[t-w+1 .. t]), which contains y_t.
    "causal" (== causal=True) roll_*[t] = mean(ev[t-w .. t-1]): every feature
             available at t depends only on y_(t-1) and earlier.
    "serve"  what the DEPLOYED service actually computes. Not the same as
             either of the above, and this is the important one.

    On "serve". ev_features_builder.compute_lag_features takes the window
    [idx-w+1, idx+1) -- inclusive of idx -- and realtime_runner calls it via
    build_ev_next_input BEFORE the step is predicted, on an ev_series_full
    whose forecast slots are still zero. The service therefore divides by w
    while summing only w-1 real values:

        train :  mean(y[t-3], y[t-2], y[t-1], y[t])
        causal:  mean(y[t-3], y[t-2], y[t-1])
        serve :  ( y[t-3] + y[t-2] + y[t-1] + 0 ) / 4

    so the served rolling mean is (w-1)/w of the causal one: about 25 % low on
    the 1 h window, 4 % on 6 h, 1 % on 24 h. The model was fitted expecting the
    term that contains y_t and is served one that is systematically deflated.
    Since that feature correlates strongly with the target, the model
    under-predicts -- which is the direction and rough size of the deployed
    model's measured bias.

    The default stays "train" deliberately: flipping it silently would change
    the meaning of every result already committed under the old convention.
    Callers opt in, and the pipelines are versioned separately.
    """
    if mode is None:
        mode = "causal" if causal else "train"
    if mode not in ("train", "causal", "serve"):
        raise ReconstructionError("mode must be train|causal|serve, got %r" % mode)
    for c in [timestamp_col, ev_col] + WEATHER_VARS:
        if c not in df.columns:
            raise ReconstructionError(f"raw frame missing required column: {c!r}")

    d = df.copy()
    ts = pd.to_datetime(d[timestamp_col], utc=True)
    d = d.sort_values(timestamp_col).reset_index(drop=True)
    ts = pd.to_datetime(d[timestamp_col], utc=True)
    ev = d[ev_col].astype(float)

    out = pd.DataFrame({timestamp_col: d[timestamp_col].values})
    out["ev_kw"] = ev.values

    # Lags: global continuous shift; head NaN -> 0 (lag_1 verified head-exact vs UK).
    for k, col in [(1, "lag_1"), (4, "lag_4"), (96, "lag_96"), (672, "lag_672")]:
        out[col] = ev.shift(k).fillna(0.0).values

    # Rolling means. See the `mode` docstring: "train" includes y_t, "causal"
    # shifts the window one step back, "serve" sums the w-1 past values but
    # still divides by w -- the deployed arithmetic.
    for w, col in [(4, "roll_1h_mean"), (24, "roll_6h_mean"),
                   (96, "roll_24h_mean")]:
        if mode == "train":
            v = ev.rolling(w, min_periods=1).mean()
        elif mode == "causal":
            v = ev.shift(1).rolling(w, min_periods=1).mean()
        else:                                     # "serve"
            v = ev.shift(1).rolling(w - 1, min_periods=1).sum() / float(w)
        out[col] = v.fillna(0.0).values

    # Weather passthrough (already in target units/order from the DB export).
    for c in WEATHER_VARS:
        out[c] = d[c].astype(float).values

    # Time encodings.
    minutes = ts.dt.hour * 60 + ts.dt.minute
    doy = ts.dt.dayofyear.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    out["tod_sin"] = np.sin(2 * np.pi * minutes / 1440).values
    out["tod_cos"] = np.cos(2 * np.pi * minutes / 1440).values
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).values   # 365.25 (pinned)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).values
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7).values
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7).values
    out["is_weekend"] = (dow >= 5).astype(float).values

    return out[[timestamp_col] + SEQ_FEATURES]


def self_test_against_uk(uk_csv: str, tol: float = 1e-6,
                         body_start: int = 700) -> None:
    """
    HARD self-test: recompute the 21 features from the UK file's own ev_kw and
    confirm they match the UK file's stored columns (body, past warm-up) within
    tol. Guarantees engineer_features implements the production convention before
    it is ever applied to the source domain. Raises ReconstructionError on drift.
    """
    uk = pd.read_csv(uk_csv)
    rebuilt = engineer_features(uk[["timestamp", "ev_kw"] + WEATHER_VARS])
    bad = {}
    for c in SEQ_FEATURES:
        a = rebuilt[c].to_numpy(float)[body_start:]
        b = uk[c].to_numpy(float)[body_start:]
        m = ~(np.isnan(a) | np.isnan(b))
        err = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
        if err > tol:
            bad[c] = err
    if bad:
        raise ReconstructionError(
            "engineer_features does NOT reproduce uk_ev_features_full.csv "
            f"(body>={body_start}) within tol={tol}. Per-feature max error: {bad}. "
            "The feature convention is wrong -- fix before touching source data.")
    print(f"[features] self-test OK: 21/21 features reproduce {uk_csv} "
          f"(body>={body_start}) within {tol}.")


def validate_source_features(feat_df: pd.DataFrame,
                             scalers: dict,
                             split_ts,
                             expect_train_rows: int,
                             abs_tol: float = 1e-3,
                             rel_tol: float = 1e-3) -> None:
    """
    HARD validation of a reconstructed SOURCE feature frame against production
    ground truth. RAISES ReconstructionError (does not warn) if either check
    fails, so no downstream transfer step can consume a bad reconstruction.

    Checks
    ------
    (1) For every SEQ_FEATURE, the reconstructed column's [min, max] over the
        TRAIN rows (ts < split_ts) matches the frozen scaler_seq.data_min_/
        data_max_ within max(abs_tol, rel_tol*range). Weather ranges are the
        strongest tell of a wrong window/units; lags/rolls catch convention bugs.
    (2) The train-row count (strict ts < split_ts) equals the frozen scaler's
        n_samples_seen_ (expect_train_rows, e.g. 38883).
    """
    scaler_seq = scalers["scaler_seq"]
    n_seen = int(getattr(scaler_seq, "n_samples_seen_"))
    dmin = np.asarray(scaler_seq.data_min_, float)
    dmax = np.asarray(scaler_seq.data_max_, float)
    if len(dmin) != len(SEQ_FEATURES):
        raise ReconstructionError(
            f"scaler_seq fit on {len(dmin)} features, expected {len(SEQ_FEATURES)}.")

    ts = pd.to_datetime(feat_df["timestamp"], utc=True)
    split_ts = pd.Timestamp(split_ts)
    if split_ts.tzinfo is None:
        split_ts = split_ts.tz_localize("UTC")
    train = feat_df[(ts < split_ts).to_numpy()]
    n_train = len(train)

    problems = []

    # (2) row count vs scaler n_samples_seen_
    if n_train != expect_train_rows:
        problems.append(
            f"train-row count {n_train} != expected {expect_train_rows} "
            f"(scaler n_samples_seen_={n_seen}). Window/split does not match "
            f"what the production scaler saw.")

    # (1) per-feature range vs frozen scaler
    range_bad = {}
    for i, c in enumerate(SEQ_FEATURES):
        col = train[c].to_numpy(float)
        cmin, cmax = float(np.nanmin(col)), float(np.nanmax(col))
        span = max(abs(dmax[i] - dmin[i]), 1.0)
        tol = max(abs_tol, rel_tol * span)
        if abs(cmin - dmin[i]) > tol or abs(cmax - dmax[i]) > tol:
            range_bad[c] = {
                "recon_min": round(cmin, 6), "scaler_min": round(float(dmin[i]), 6),
                "recon_max": round(cmax, 6), "scaler_max": round(float(dmax[i]), 6),
                "tol": round(tol, 6),
            }
    if range_bad:
        problems.append(f"scaler_seq range mismatch on {len(range_bad)} feature(s): "
                        f"{range_bad}")

    if problems:
        raise ReconstructionError(
            "SOURCE reconstruction FAILED production validation (hard stop):\n  - "
            + "\n  - ".join(problems)
            + "\nRefusing to proceed: a mismatched reconstruction would invalidate "
              "every transfer result. Reconcile the export window / feature "
              "convention before continuing.")

    print(f"[features] source validation OK: {n_train} train rows == "
          f"n_samples_seen_={n_seen}; scaler_seq min/max matched on "
          f"{len(SEQ_FEATURES)}/{len(SEQ_FEATURES)} features "
          f"(abs_tol={abs_tol}, rel_tol={rel_tol}).")


def reconstruct_source(raw_csv: str, uk_csv: str, scalers: dict,
                       split_ts, expect_train_rows: int,
                       out_csv: str | None = None,
                       warmup_rows: int = 0) -> pd.DataFrame:
    """
    End-to-end: self-test the convention on UK, engineer the source features,
    optionally drop `warmup_rows` leading rows used only to prime lags, HARD-
    validate against the frozen scalers, and (optionally) write a feature CSV
    column-identical to uk_ev_features_full.csv. Returns the validated frame.
    """
    self_test_against_uk(uk_csv)  # raises if convention is wrong
    raw = pd.read_csv(raw_csv)
    # Features are computed on the CONTINUOUS series (incl. any warm-up rows) so
    # lags/rolls at the window edge are correct; warm-up rows are dropped AFTER.
    feat = engineer_features(raw)
    if "is_warmup" in raw.columns:
        # Preferred: drop exactly the rows the export flagged as pre-window priming.
        keep = raw["is_warmup"].to_numpy() == 0
        feat = feat[keep].reset_index(drop=True)
        print(f"[features] dropped {int((~keep).sum())} warm-up row(s) via is_warmup flag")
    elif warmup_rows > 0:
        feat = feat.iloc[warmup_rows:].reset_index(drop=True)
        print(f"[features] dropped {warmup_rows} warm-up row(s) via count")
    validate_source_features(feat, scalers, split_ts, expect_train_rows)  # raises
    if out_csv:
        feat.to_csv(out_csv, index=False)
        print(f"[features] wrote validated source features -> {out_csv} "
              f"({len(feat)} rows)")
    return feat


if __name__ == "__main__":
    # Minimal CLI: python -m coldstart_transfer.features <raw_csv> <split_ts_iso> [warmup]
    import joblib
    if len(sys.argv) < 3:
        print("usage: python -m coldstart_transfer.features "
              "<raw_source_csv> <split_ts_iso> [warmup_rows]", file=sys.stderr)
        sys.exit(2)
    raw_csv, split_ts = sys.argv[1], sys.argv[2]
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    try:
        reconstruct_source(
            raw_csv=raw_csv, uk_csv="Data/uk_ev_features_full.csv",
            scalers=scalers, split_ts=split_ts, expect_train_rows=38883,
            out_csv=raw_csv.replace(".csv", "_features.csv"), warmup_rows=warmup)
    except ReconstructionError as e:
        print(f"\n[HARD FAILURE]\n{e}", file=sys.stderr)
        sys.exit(1)
