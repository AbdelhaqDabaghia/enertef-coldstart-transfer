#!/bin/bash
# GPU-enabled runner for the cold-start pipeline in WSL.
# Usage: wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/dev/service1/scripts/wsl_run.sh <python-args>
# e.g.  ... wsl_run.sh -m coldstart_transfer.model Data/models/ev_cnn_lstm_20260718.keras
VENV=/root/tfgpu
NVBASE=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(ls -d $NVBASE/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
cd /mnt/c/dev/service1
export PYTHONPATH=/mnt/c/dev/service1:${PYTHONPATH:-}
exec $VENV/bin/python "$@"
