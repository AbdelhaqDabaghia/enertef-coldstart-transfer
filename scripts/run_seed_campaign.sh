#!/bin/bash
# run_seed_campaign.sh -- re-run the formerly-3-seed CL experiments at N seeds
# (default 10) for the statistical-robustness revision, then emit paper-ready
# stats (95% CI + paired Wilcoxon + Holm + TOST equivalence) via stats_report.
#
# WHY: reviewers flag that the central EWC/replay/warm-start tables rest on 3
# seeds (CV up to ~35%). This bumps every such table to 10 seeds and, crucially,
# tests the negative "EWC adds nothing" claim with TOST rather than a
# non-significant difference test.
#
# Run on the GPU laptop, from the repo root, through the WSL GPU env, e.g.:
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/dev/<repo>/scripts/run_seed_campaign.sh
# or with an activated venv:  bash scripts/run_seed_campaign.sh
#
# Config via env:
#   SEEDS   seed list (default 0..9)          PY   python to use (default python)
#   TOST_T  target-nRMSE margin (default .02) TOST_S retention margin (default .05)
#   SKIP_RUN=1  only (re)generate stats from existing CSVs, no GPU training
set -euo pipefail

# --- GPU env (mirrors scripts/wsl_run.sh) -------------------------------------
# Set up CUDA libs + PYTHONPATH so TensorFlow sees the GPU. Skip entirely by
# passing NO_GPU_ENV=1 (e.g. inside an already-activated venv, or SKIP_RUN=1).
#   VENV   path to the tf-gpu venv (default /root/tfgpu)
VENV="${VENV:-/root/tfgpu}"
if [[ "${NO_GPU_ENV:-0}" != "1" && -d "$VENV" ]]; then
  NVBASE="$VENV/lib/python3.12/site-packages/nvidia"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(ls -d $NVBASE/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
  export TF_CPP_MIN_LOG_LEVEL=2
  export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
  PY="${PY:-$VENV/bin/python}"
fi

SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
PY="${PY:-python}"
TOST_T="${TOST_T:-0.02}"
TOST_S="${TOST_S:-0.05}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RES="Data/results"
ARCH="$RES/_pre_campaign_$STAMP"
STATS="$RES/stats"
mkdir -p "$STATS"

echo "[campaign] SEEDS=$SEEDS  PY=$PY  target-δ=$TOST_T  retention-δ=$TOST_S"

# (module -> output CSV) that were run at 3 seeds and feed a central table.
# NOTE: the full CL-taxonomy table (MAS/LwF/DER++/A-GEM) has NO driver in THIS
# repo -- confirm where it is produced before claiming 10 seeds for it.
declare -a JOBS=(
  "coldstart_transfer.driver            rq1_coldstart.csv"        # tab:rq1  (RQ1)
  "coldstart_transfer.driver_source     rq2_rq3_source.csv"       # tab:rq2  (RQ2/RQ3)
  "coldstart_transfer.driver_ewc_bench  ewc_bench.csv"            # tab:bench
  "coldstart_transfer.driver_fisher_control rq3_fisher_control.csv" # tab:rq3
  "coldstart_transfer.driver_pv         pv_rq1_coldstart.csv"     # PV tab:pv RQ1
  "coldstart_transfer.driver_pv_source  pv_rq2_rq3_source.csv"    # PV tab:pv RQ2/RQ3
)

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
  mkdir -p "$ARCH"
  echo "[campaign] archiving existing CSVs -> $ARCH (fresh uniform 10-seed files)"
  for j in "${JOBS[@]}"; do
    csv="$RES/$(echo "$j" | awk '{print $2}')"
    [[ -f "$csv" ]] && mv "$csv" "$ARCH/" && echo "  archived $(basename "$csv")"
  done
  for j in "${JOBS[@]}"; do
    mod="$(echo "$j" | awk '{print $1}')"
    echo "===================================================================="
    echo "[campaign] $mod  (SEEDS=$SEEDS)"
    echo "===================================================================="
    SEEDS="$SEEDS" "$PY" -m "$mod"
  done
fi

echo "[campaign] generating statistics -> $STATS/"
# EWC bench: plasticity (target) + stability (retention), grouped by variant name
"$PY" -m coldstart_transfer.stats_report --csv "$RES/ewc_bench.csv" \
  --group notes --ref B3_warm --metric target_nrmse --tost-margin "$TOST_T" \
  --out "$STATS/ewc_bench_target.md"
"$PY" -m coldstart_transfer.stats_report --csv "$RES/ewc_bench.csv" \
  --group notes --ref B3_warm --metric source_retention_nrmse --tost-margin "$TOST_S" \
  --out "$STATS/ewc_bench_retention.md"
# RQ2/RQ3 conditions (B0/B3/B1/B4/B5) keyed by the baseline column
"$PY" -m coldstart_transfer.stats_report --csv "$RES/rq2_rq3_source.csv" \
  --group baseline --ref B3 --metric target_nrmse --tost-margin "$TOST_T" \
  --out "$STATS/rq2_target.md"
"$PY" -m coldstart_transfer.stats_report --csv "$RES/rq2_rq3_source.csv" \
  --group baseline --ref B3 --metric source_retention_nrmse --tost-margin "$TOST_S" \
  --out "$STATS/rq2_retention.md"

echo "[campaign] DONE. Stats markdown in $STATS/  (archived raw in $ARCH)"
