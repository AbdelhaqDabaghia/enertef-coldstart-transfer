"""
logger.py -- append-only results logger for cold-start transfer runs.

One row per run capturing everything needed for RQ1/RQ2/RQ3 analysis and honest
variance reporting: baseline, N days, lambda, beta, target nRMSE, source-
retention nRMSE, data-to-threshold flag, wall-clock, seed, timestamp.
"""
from __future__ import annotations
import os
import csv
from datetime import datetime, timezone

FIELDS = [
    "run_id", "baseline", "n_days", "lambda_ewc", "beta_replay", "seed",
    "target_nrmse", "target_rmse_kw", "target_mean_kw",
    "source_retention_nrmse", "below_threshold", "threshold",
    "epochs", "batch_size", "lr", "wall_clock_s", "timestamp", "notes",
]


def log_result(path: str, row: dict) -> None:
    """Append one result row to `path` (creates header if new)."""
    row = dict(row)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
