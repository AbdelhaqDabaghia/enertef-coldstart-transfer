"""
build_causal_features.py -- rebuild both feature files with causal rolling means.

E7. Every result in this repository was computed on features whose trailing
rolling means include the current step, so roll_*[t] contains ev[t], and
windowing hands exactly that vector to the model as X_next at the predicted
step. m4_leakage_sensitivity.csv measures the cost of removing it: source
holdout 0.627 -> 0.895, target zero-shot 0.993 -> 1.445.

This script rebuilds uk_ev_features_full.csv and lux_source_features.csv from
their own ev_kw and weather columns with causal=True, writing them alongside
under *_causal.csv. Row counts, ordering and timestamps are preserved exactly,
so every downstream window index is unchanged and the two pipelines are
comparable row for row.

SCALERS. The frozen production scalers stay in use and are NOT re-fit. They
were fitted on the leaky convention, so their per-feature min/max are now
slightly off for the three rolling columns. That is deliberate: re-fitting would
break weight compatibility with the transferred production model, which is the
artefact under study. We report the induced range mismatch below so the size of
the compromise is on the record rather than assumed negligible.

Writes Data/uk_ev_features_causal.csv and Data/lux_source_features_causal.csv.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd

from coldstart_transfer.features import (engineer_features, SEQ_FEATURES,
                                         WEATHER_VARS)

PAIRS = [("Data/uk_ev_features_full.csv", "Data/uk_ev_features_causal.csv"),
         ("Data/lux_source_features.csv", "Data/lux_source_features_causal.csv")]
ROLLS = ["roll_1h_mean", "roll_6h_mean", "roll_24h_mean"]


def main():
    scalers = joblib.load("Data/models/ev_scalers.joblib")
    sq = scalers["scaler_seq"]
    dmin = np.asarray(sq.data_min_, float)
    dmax = np.asarray(sq.data_max_, float)

    for src, dst in PAIRS:
        df = pd.read_csv(src)
        raw = df[["timestamp", "ev_kw"] + WEATHER_VARS]
        causal = engineer_features(raw, causal=True)

        assert len(causal) == len(df), "row count changed -- window indices would shift"
        assert (causal["timestamp"].to_numpy() == df["timestamp"].to_numpy()).all(), \
            "timestamps changed"
        assert list(causal.columns) == ["timestamp"] + SEQ_FEATURES, "column order changed"

        # how far the causal build moves each feature, and how far it now sits
        # outside the frozen scaler's fitted range
        print("\n=== %s ===" % src)
        print("  %-16s %10s %10s %10s" % ("feature", "max|delta|", "min_off", "max_off"))
        for c in ROLLS:
            i = SEQ_FEATURES.index(c)
            a = df[c].to_numpy(float)
            b = causal[c].to_numpy(float)
            span = max(dmax[i] - dmin[i], 1e-9)
            print("  %-16s %10.3f %9.2f%% %9.2f%%"
                  % (c, np.max(np.abs(a - b)),
                     100 * (dmin[i] - b.min()) / span,
                     100 * (b.max() - dmax[i]) / span))

        # the target column itself must be untouched -- if ev_kw moved, the
        # comparison between pipelines would not be like for like
        assert np.allclose(df["ev_kw"].to_numpy(float),
                           causal["ev_kw"].to_numpy(float)), "ev_kw changed"

        causal.to_csv(dst, index=False)
        print("  wrote %s (%d rows)" % (dst, len(causal)))

    print("\n[build_causal] done. Point UK_CSV / LUX_CSV at the *_causal.csv "
          "files to run an experiment on the causal pipeline.")


if __name__ == "__main__":
    main()
