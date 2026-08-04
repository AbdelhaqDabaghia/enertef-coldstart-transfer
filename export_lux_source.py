"""
export_lux_source.py  --  READ-ONLY export of the Luxembourg source-domain
window that the 2026-07-18 EV EWC run trained on.

Run this on YOUR side, where SSH/PG access works (this Claude Code session is
behind a DPI middlebox that drops the SSH banner exchange, so it cannot reach
the bastion). Start the tunnel first, then run:

    ssh -i C:\\dev\\bastion-host.pem -o ServerAliveInterval=30 \
        -L 5433:enertef-postgres.chqenv2wr2hv.eu-central-1.rds.amazonaws.com:5432 \
        ec2-user@3.74.159.169 -N

    set PG_HOST=localhost & set PG_PORT=5433 & set PG_DB=enertef ^
      & set PG_USER=enertef_admin & set PG_PASSWORD=...        (do NOT hard-code)
    python export_lux_source.py

SAFETY: every statement runs inside a READ ONLY transaction. No INSERT/UPDATE/
DELETE/DDL. This DB serves a live 15-min MPC control loop -- read only.

Output: Data/lux_source_raw_window.csv with columns
    timestamp, ev_kw, shortwave_radiation, direct_radiation, diffuse_radiation,
    cloud_cover, temperature_2m, wind_speed_10m
plus Data/lux_source_window_provenance.json recording the exact window bounds,
row counts, and the job-time cap used -- so the experiment's source slice is
fully reproducible and auditable.

Feature engineering (the 21 SEQ_FEATURES + the 7-day holdout split) is done
separately, on this raw window, by the reconstruction step that validates
against the frozen production scalers (ev_kw in [0, 578.4]).
"""
import os
import csv
import json
import ssl
from datetime import timedelta

# We export the FULL restricted EMOB2-active window (from the first timestamp
# where EMOB2 is present, through the capped latest-common timestamp), NOT just
# a trailing slice.  This lets the feature-engineering step re-derive the exact
# rows the frozen production scaler saw and re-validate n_samples_seen_ == 38883.
#
# EMPIRICALLY VERIFIED against local ECC_master (weather-independent, so checkable
# offline): restricting to EMOB2-active rows (47,524) and splitting at
# split_ts = max_ts - 90d with a STRICT `<` train side yields exactly 38,883
# training rows == the frozen scaler's n_samples_seen_.  This is the authoritative
# ground truth; the DFL repo's fillna(0)-over-full-history path is a DIFFERENT
# design and is deliberately NOT used here.
#
# CAP CHOICE: to reproduce the scaler-fit distribution set JOB_TIME_CAP to the
# scaler-freeze data end (~2026-01-19; gives the 47,524-row window ending
# 2026-01-18 22:45 UTC).  To instead capture the window the 2026-07-18 EWC weights
# were fine-tuned on, set JOB_TIME_CAP="2026-07-18 07:45:00+00".  These are two
# DIFFERENT windows (see handoff note); pick per what the run needs.
JOB_TIME_CAP = os.environ.get("JOB_TIME_CAP", "2026-01-19 00:00:00+00")
# Ingestion cap: historical.telemetry_15min has a created_at (tstz) distinct from
# timestamp. Bounding BOTH timestamp <= cap AND created_at <= cap gives an EXACT
# point-in-time snapshot of what a job saw when it started (not an approximation),
# because it excludes any rows backfilled AFTER the job ran.
#   - Window B (07-18 EWC): CREATED_AT_CAP = JOB_TIME_CAP = 2026-07-18 07:45Z  -> exact.
#   - Window A (scaler-freeze): the Jan-2026 timestamps may have been re-ingested
#     later, so set CREATED_AT_CAP to the scaler-freeze time (~2026-06-30) or leave
#     it generous; the built-in 38,883 self-check verifies the window is right.
# Defaults to JOB_TIME_CAP (exact-snapshot semantics) unless overridden.
CREATED_AT_CAP = os.environ.get("CREATED_AT_CAP", JOB_TIME_CAP)
TEST_DAYS = int(os.environ.get("TEST_DAYS", "90"))       # notebook split: max - 90d
HOLDOUT_DAYS = int(os.environ.get("HOLDOUT_DAYS", "7"))  # gate holdout (13.58/42.04 kW)
EXPECT_TRAIN_ROWS = int(os.environ.get("EXPECT_TRAIN_ROWS", "38883"))  # scaler ground truth

# Optional warm-up: fetch WARMUP_ROWS extra 15-min steps BEFORE window_start so
# the feature step can compute correct lag_672/roll_24h at the window's leading
# edge, then drop them (features.reconstruct_source(warmup_rows=...)). These rows
# are NOT part of the validated window and are EXCLUDED from the 38,883 count.
# Use >= 672 (one full lookback) if window-A validation trips on head lag/roll
# features; leave 0 if production built features on the isolated restricted window
# (head lags zero-filled) -- the hard scaler validation decides which is correct.
WARMUP_ROWS = int(os.environ.get("WARMUP_ROWS", "0"))

OUT_CSV = os.path.join("Data", "lux_source_raw_window.csv")
OUT_PROV = os.path.join("Data", "lux_source_window_provenance.json")

WX_SELECT = ("ghi_w_m2, dni_w_m2, dhi_w_m2, cloud_pct, temp_celsius, wind_speed_10m")
WX_KEYS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "cloud_cover", "temperature_2m", "wind_speed_10m"]


def get_conn():
    """Prefer pg8000 (matches production data_loader.py); fall back to psycopg2."""
    host = os.environ.get("PG_HOST", "localhost")
    port = int(os.environ.get("PG_PORT", "5433"))
    db = os.environ.get("PG_DB", "enertef")
    user = os.environ.get("PG_USER", "enertef_admin")
    pw = os.environ.get("PG_PASSWORD", "")
    if not pw:
        raise SystemExit("Set PG_PASSWORD in the environment (do not hard-code it).")
    try:
        import pg8000.dbapi
        pg8000.dbapi.paramstyle = "format"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return pg8000.dbapi.connect(host=host, port=port, database=db,
                                    user=user, password=pw, ssl_context=ctx)
    except ImportError:
        import psycopg2
        return psycopg2.connect(host=host, port=port, dbname=db,
                                user=user, password=pw, sslmode="require")


def main():
    conn = get_conn()
    cur = conn.cursor()
    # Hard read-only guard for the whole session.
    cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")

    # 1) end = get_latest_common_timestamp(['EMOB1','EMOB2']) as of the job time.
    #    Point-in-time: bound BOTH the data timestamp AND the ingestion time
    #    (created_at) so backfills after the job ran cannot leak in -> EXACT.
    cur.execute(
        """
        SELECT MIN(latest_ts) FROM (
            SELECT asset_id, MAX(timestamp) AS latest_ts
            FROM historical.telemetry_15min
            WHERE asset_id IN ('EMOB1','EMOB2')
              AND timestamp <= %s AND created_at <= %s
            GROUP BY asset_id
        ) t
        """, (JOB_TIME_CAP, CREATED_AT_CAP))
    end = cur.fetchone()[0]
    if end is None:
        raise SystemExit("No EMOB1/EMOB2 data at/before caps (timestamp & created_at).")

    # start = first timestamp where EMOB2 is present (restricted window's left
    # edge), under the same point-in-time ingestion bound.
    cur.execute(
        """SELECT MIN(timestamp) FROM historical.telemetry_15min
           WHERE asset_id='EMOB2' AND created_at <= %s""", (CREATED_AT_CAP,))
    start = cur.fetchone()[0]

    split_ts = end - timedelta(days=TEST_DAYS)        # notebook train/test boundary
    holdout_start = end - timedelta(days=HOLDOUT_DAYS)  # gate holdout (13.58/42.04 kW)
    # Warm-up lower bound: WARMUP_ROWS steps of 15 min before the true window edge.
    fetch_start = start - timedelta(minutes=15 * WARMUP_ROWS)
    print(f"[WINDOW] restricted EMOB2-active: start={start}  end={end}")
    print(f"[SPLIT]  split_ts={split_ts} (train = start <= ts < split_ts, strict)")
    print(f"[HOLD]   holdout_start={holdout_start}")
    if WARMUP_ROWS:
        print(f"[WARMUP] fetching {WARMUP_ROWS} pre-window steps from {fetch_start} "
              f"(dropped by the feature step; excluded from the {EXPECT_TRAIN_ROWS} count)")

    # 2) EV = EMOB1 + EMOB2 (fillna 0), over [fetch_start, end] (incl. warm-up).
    def fetch_asset(asset):
        # Same point-in-time ingestion bound so every asset reflects the DB state
        # as of the job time, not today's (possibly backfilled) state.
        cur.execute(
            """SELECT timestamp, power_kw_avg FROM historical.telemetry_15min
               WHERE asset_id=%s AND timestamp>=%s AND timestamp<=%s
                 AND created_at <= %s
               ORDER BY timestamp""", (asset, fetch_start, end, CREATED_AT_CAP))
        return {r[0]: float(r[1] or 0.0) for r in cur.fetchall()}

    e1 = fetch_asset("EMOB1")
    e2 = fetch_asset("EMOB2")
    all_ts = sorted(set(e1) | set(e2))
    ev = {t: e1.get(t, 0.0) + e2.get(t, 0.0) for t in all_ts}
    print(f"[EV] {len(ev)} pts (EMOB1={len(e1)}, EMOB2={len(e2)})")

    # 3) Weather over [start, end]. NOTE: contextual.weather has no confirmed
    #    created_at column, so it is NOT ingestion-bounded. Weather is exogenous
    #    and effectively static per timestamp (backfilled once), so this does not
    #    affect the point-in-time fidelity of the EV target/window.
    cur.execute(
        f"""SELECT timestamp, {WX_SELECT} FROM contextual.weather
            WHERE timestamp>=%s AND timestamp<=%s ORDER BY timestamp""",
        (fetch_start, end))
    wx = {r[0]: [float(x or 0.0) for x in r[1:]] for r in cur.fetchall()}
    print(f"[WX] {len(wx)} pts")

    conn.rollback()  # read-only; end txn without writing
    cur.close(); conn.close()

    # 4) Write raw window CSV (feature engineering happens later, validated vs
    #    scalers). `is_warmup`=1 marks the pre-window priming rows (t < start).
    os.makedirs("Data", exist_ok=True)
    n_warmup = sum(1 for t in all_ts if t < start)
    zero_wx = [0.0] * len(WX_KEYS)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "ev_kw"] + WX_KEYS + ["is_warmup"])
        for t in all_ts:
            w.writerow([t.isoformat(), ev[t]] + wx.get(t, zero_wx)
                       + [1 if t < start else 0])
    print(f"[OUT] wrote {OUT_CSV} ({len(all_ts)} rows, {n_warmup} warm-up)")

    # Self-check: does the restricted window + strict-< split reproduce the frozen
    # scaler's n_samples_seen_ (38,883)?  Warm-up rows (t < start) are EXCLUDED --
    # only [start, split_ts) count as train.  (Weather-independent row/ts logic.)
    train_rows = sum(1 for t in all_ts if start <= t < split_ts)
    scaler_match = (train_rows == EXPECT_TRAIN_ROWS)
    print(f"[VALIDATE] train_rows(start<=ts<split)={train_rows}  expected(scaler "
          f"n_samples_seen_)={EXPECT_TRAIN_ROWS}  MATCH={scaler_match}")
    if not scaler_match:
        print("[VALIDATE][WARN] train-row count != frozen scaler count. The cap or "
              "window definition does not match what the production scaler saw -- "
              "do NOT proceed to feature reconstruction until reconciled.")

    # created_at bounding makes this an EXACT point-in-time snapshot (backfills
    # after the job cannot leak in), not an approximation.
    reproduction = ("exact (point-in-time: timestamp<=cap AND created_at<=cap)"
                    if CREATED_AT_CAP else "approximate (timestamp cap only)")
    prov = {
        "job_time_cap": JOB_TIME_CAP,
        "created_at_cap": CREATED_AT_CAP,
        "reproduction": reproduction,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "split_ts": split_ts.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "test_days": TEST_DAYS,
        "holdout_days": HOLDOUT_DAYS,
        "n_rows": len(all_ts),
        "warmup_rows_requested": WARMUP_ROWS,
        "n_warmup_rows": n_warmup,
        "fetch_start": fetch_start.isoformat(),
        "n_train_rows_strict_lt": train_rows,
        "expected_scaler_n_samples_seen": EXPECT_TRAIN_ROWS,
        "scaler_row_count_match": scaler_match,
        "n_emob1": len(e1),
        "n_emob2": len(e2),
        "n_weather": len(wx),
        "weather_ingestion_bounded": False,
        "ev_kw_min": min(ev.values()) if ev else None,
        "ev_kw_max": max(ev.values()) if ev else None,
        "note": ("Restricted EMOB2-active window (NOT full-history fillna(0)); "
                 "train side uses strict ts < split_ts. EV telemetry is bounded on "
                 "BOTH timestamp and created_at for an exact point-in-time snapshot. "
                 "Window B (JOB_TIME_CAP=CREATED_AT_CAP=2026-07-18 07:45Z) exactly "
                 "reproduces the 07-18 EWC run's view. Window A (caps at scaler "
                 "freeze) reproduces the frozen scaler's n_samples_seen_=38883; "
                 "if the Jan-2026 rows were re-ingested later, raise CREATED_AT_CAP "
                 "for window A until scaler_row_count_match is true."),
    }
    with open(OUT_PROV, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[OUT] wrote {OUT_PROV}")
    print(f"[CHECK] ev_kw range = [{prov['ev_kw_min']}, {prov['ev_kw_max']}] "
          f"(production scaler_y was fit on [0.0, 578.4])")


if __name__ == "__main__":
    main()
