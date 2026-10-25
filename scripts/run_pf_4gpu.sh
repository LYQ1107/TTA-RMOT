#!/usr/bin/env bash
set -euo pipefail
# Dry by default; up to four actually idle GPUs, never occupied-card bypass.
PF_REPO="${PF_REPO:-/data2/user/FlexHook-TTA}"
PF_PYTHON="${PF_PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}"
export PYTHONPATH="$PF_REPO/compat:$PF_REPO:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}"
exec "$PF_PYTHON" "$PF_REPO/scripts/launch_pf_shards.py" "$@"
