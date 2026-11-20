#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-7}"
OUTPUT_DIR="${TTA_RMOT_V2_RUN_ROOT:-/data3/user/TTA-RMOT-runs/shift_rmot_v2_independent}"
CONFIG_PATH="${TTA_RMOT_V2_CONFIG:-${ROOT}/configs/tta/shift_rmot_v2_independent.yaml}"
PYTHONPATH_VALUE="${PYTHONPATH:-}"
if [[ -n "${PYTHONPATH_VALUE}" ]]; then
  PYTHONPATH_VALUE="${ROOT}:${PYTHONPATH_VALUE}"
else
  PYTHONPATH_VALUE="${ROOT}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="/data2/user/runtime_site_locatemot:${PYTHONPATH_VALUE}"

exec python3 "${ROOT}/scripts/run_target_suite.py" \
  --config "${CONFIG_PATH}" \
  --runtime-manifest "${ROOT}/data/ShiftRMOT-v2/manifests/paired_conditions.jsonl" \
  --output-dir "${OUTPUT_DIR}" "$@"
