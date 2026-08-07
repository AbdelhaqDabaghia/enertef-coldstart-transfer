#!/bin/bash
# Chain the EV 8-seed podium run then the PV ablation, sequentially, in one process.
# Both drivers are incremental + resume-safe, so relaunching this script after any
# kill continues where it stopped. Does NOT use exec (which would prevent the second).
VENV=/root/tfgpu
NVBASE=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(ls -d $NVBASE/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
cd /mnt/c/dev/service1 || exit 1
export PYTHONPATH=/mnt/c/dev/service1:${PYTHONPATH:-}

echo "=== [1/2] EV 8-seed podium (revin,per_unit,robust,zscore) ==="
ABL_SEEDS=8 ABL_REPS=revin,per_unit,robust,zscore \
  ABL_OUT=Data/results/representation_ablation_ev_8seed.csv \
  $VENV/bin/python scripts/driver_representation_ablation.py
echo "=== [1/2] EV 8-seed exit=$? ==="

echo "=== [2/2] PV ablation (clear_sky vs revin vs global) ==="
ABL_SEEDS=5 \
  ABL_OUT=Data/results/representation_ablation_pv.csv \
  $VENV/bin/python scripts/driver_representation_ablation_pv.py
echo "=== [2/2] PV exit=$? ==="
echo "=== CHAIN DONE ==="
