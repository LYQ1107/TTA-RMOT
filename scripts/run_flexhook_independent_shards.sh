#!/usr/bin/env bash
set -euo pipefail

# Run the released FlexHook test loader as independent sampler shards.  This
# keeps the two ranks from blocking on NCCL after inference and is useful when
# another workload makes one GPU finish much later than the other.
REPO=${REPO:-/data2/user/FlexHook-TTA}
VENDOR="$REPO/vendor/FlexHook"
DATA_ROOT=${DATA_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance}
TRACK_ROOT=${TRACK_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a FlexHook second-stage checkpoint}
OUT=${OUT:?set OUT to a fresh run directory}
GPUS=${GPUS:-3,5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-240}
PYTHON=${PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}

IFS=, read -r GPU0 GPU1 <<< "$GPUS"
test -n "$GPU0"
test -n "$GPU1"
test -f "$CHECKPOINT"
test -d "$DATA_ROOT"
test -d "$TRACK_ROOT"
mkdir -p "$OUT"

run_rank() {
  local rank="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$REPO/compat:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}" \
  OMP_NUM_THREADS=1 \
  "$PYTHON" "$REPO/scripts/recover_flexhook_rank_result.py" \
    --vendor "$VENDOR" \
    --cfg "$VENDOR/configs/infer/dance.yaml" \
    --data-root "$DATA_ROOT" \
    --track-root "$TRACK_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUT" \
    --rank "$rank" \
    --world-size 2 \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --log-rank "$rank" \
    > "$OUT/recovery_rank${rank}.stdout.log" 2>&1
}

run_rank 0 "$GPU0" & PID0=$!
run_rank 1 "$GPU1" & PID1=$!

status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?
exit "$status"
