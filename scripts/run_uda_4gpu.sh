#!/usr/bin/env bash
set -euo pipefail
UDA_REPO="${UDA_REPO:-/data2/user/FlexHook-TTA}"
UDA_PYTHON="${UDA_PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}"
export PYTHONPATH="$UDA_REPO/compat:$UDA_REPO:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}"
exec "$UDA_PYTHON" "$UDA_REPO/scripts/launch_uda_workers.py" "$@"
