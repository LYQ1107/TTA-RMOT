#!/usr/bin/env bash
# Source this file for the restored project's isolated Python environment.
tta_a100_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bash "$tta_a100_root/scripts/mount_nas_a800.sh" || return
export TTA_RMOT_ROOT="$tta_a100_root"
export TTA_RMOT_NAS=/data3/liuyeqiang/nas/A800
export TTA_RMOT_PATHS="$tta_a100_root/configs/a100_paths.json"
export TTA_RMOT_PYTHON="$tta_a100_root/.venv/bin/python"
export TTA_RMOT_OUTPUT="$tta_a100_root/a100_runs"
export PATH="$tta_a100_root/.venv/bin:$PATH"
export PYTHONPATH="$tta_a100_root${PYTHONPATH:+:$PYTHONPATH}"
unset tta_a100_root
