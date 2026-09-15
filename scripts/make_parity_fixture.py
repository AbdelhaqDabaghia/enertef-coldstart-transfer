"""
make_parity_fixture.py -- capture what the deployed builder computes.

WHY A FIXTURE RATHER THAN THE CODE. The parity test needs the deployed
ev_features_builder, which lives in a private repository. CI here is public, so
the two obvious routes both cost something: checking the private repo out needs
a personal access token in a public workflow, and vendoring the builder would
publish production code.

A golden fixture avoids both. It records what the real builder produced for a
fixed input -- numbers only, no logic -- so CI can assert our `serve` mode
still reproduces it. Regenerating the fixture requires the real builder and is
therefore a local, deliberate act.

THE OBVIOUS OBJECTION. A committed fixture is a snapshot, and a snapshot goes
stale: if the deployed builder changes, CI keeps passing against yesterday's
behaviour. That is the same class of fault as the one this project exists to
fix, so the mitigation is explicit -- tests/test_serving_parity.py regenerates
from the live builder whenever PROD_BUILDER points at it and fails on any
difference. CI proves our side did not drift; a developer run proves the
builder did not. Neither alone is sufficient and the fixture records which is
which.

    python scripts/make_parity_fixture.py
    PROD_BUILDER=/path/to/ev_features_builder.py python scripts/make_parity_fixture.py

Writes tests/fixtures/serving_parity.json.
"""
from __future__ import annotations
import os
import json
import hashlib
import importlib.util
import datetime as dt
import numpy as np

PROD_BUILDER = os.environ.get(
    "PROD_BUILDER", r"C:\dev\svc1-runner\ev_features_builder.py")
OUT = os.path.join("tests", "fixtures", "serving_parity.json")

N, SEED = 900, 20260915
IDXS = [700, 750, 800, 850]          # a few positions, all past the 672 warm-up
ROLLS = ["roll_1h_mean", "roll_6h_mean", "roll_24h_mean"]
LAGS = ["lag_1", "lag_4", "lag_96", "lag_672"]


def load_builder(path):
    spec = importlib.util.spec_from_file_location("prod_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not os.path.exists(PROD_BUILDER):
        raise SystemExit(
            "[fixture] deployed builder not found at %s.\n"
            "  This script must run where the real builder is available; that is\n"
            "  the whole point of it. Set PROD_BUILDER." % PROD_BUILDER)

    prod = load_builder(PROD_BUILDER)
    src = open(PROD_BUILDER, "rb").read()

    rng = np.random.default_rng(SEED)
    ev = rng.gamma(2.0, 20.0, N).astype(float)

    cases = []
    for idx in IDXS:
        # what production holds at prediction time: the target slot is still
        # zero, because realtime_runner writes the prediction only afterwards
        served = ev.copy()
        served[idx] = 0.0
        got = prod.compute_lag_features(served, idx)
        cases.append({"idx": idx,
                      "features": {k: float(got[k]) for k in ROLLS + LAGS}})

    fixture = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "builder_path": os.path.basename(PROD_BUILDER),
        "builder_sha256": hashlib.sha256(src).hexdigest(),
        "builder_bytes": len(src),
        "series_seed": SEED,
        "series_n": N,
        "series_dist": "gamma(shape=2.0, scale=20.0)",
        "note": ("Captured from the deployed builder. Numbers only -- no "
                 "production logic is reproduced here. Regenerate with "
                 "scripts/make_parity_fixture.py wherever the real builder is "
                 "available; tests/test_serving_parity.py checks the live "
                 "builder against this whenever PROD_BUILDER is set."),
        "cases": cases,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)

    print("[fixture] builder %s (%d bytes)" % (fixture["builder_path"],
                                               fixture["builder_bytes"]))
    print("[fixture] sha256  %s" % fixture["builder_sha256"][:16])
    print("[fixture] %d cases -> %s" % (len(cases), OUT))
    for c in cases:
        print("   idx %d  roll_1h=%.4f  lag_1=%.4f"
              % (c["idx"], c["features"]["roll_1h_mean"],
                 c["features"]["lag_1"]))


if __name__ == "__main__":
    main()
