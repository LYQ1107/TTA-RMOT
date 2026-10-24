#!/usr/bin/env bash
set -euo pipefail

# Independent whole-episode shards for the fixed 25-video Dance validation
# set.  Each child sees one physical GPU and owns complete (video, query)
# episodes, so no adapter state crosses processes.
REPO=${REPO:-/data2/user/FlexHook-TTA}
PYTHON=${PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}
VENDOR=${VENDOR:-$REPO/vendor/FlexHook}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT}
TRACKER_ROOT=${TRACKER_ROOT:?set TRACKER_ROOT}
SOURCE_RESULT=${SOURCE_RESULT:?set SOURCE_RESULT}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
METHOD=${METHOD:-PCD_RESIDUAL}
GPUS=${GPUS:-7,5,6,4}
BATCH_SIZE=${BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
UPDATE_INTERVAL=${UPDATE_INTERVAL:-4}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
SHARD_COUNT=${#GPU_LIST[@]}
if (( SHARD_COUNT < 1 || SHARD_COUNT > 4 )); then
  echo "expected 1-4 GPUs, got $SHARD_COUNT" >&2
  exit 2
fi
mapfile -t VIDEOS < <(find "$TRACKER_ROOT" -maxdepth 1 -type f -name '*.txt' -printf '%f\n' | sed 's/\.txt$//' | sort)
if (( ${#VIDEOS[@]} != 25 )); then
  echo "expected 25 Dance tracker files, got ${#VIDEOS[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
declare -a VIDEO_GROUPS
for (( shard=0; shard<SHARD_COUNT; shard++ )); do
  VIDEO_GROUPS[$shard]=""
done
for (( index=0; index<${#VIDEOS[@]}; index++ )); do
  shard=$(( index % SHARD_COUNT ))
  VIDEO_GROUPS[$shard]+="${VIDEOS[$index]} "
done

declare -a PIDS
for (( shard=0; shard<SHARD_COUNT; shard++ )); do
  read -r -a SHARD_VIDEOS <<< "${VIDEO_GROUPS[$shard]}"
  SHARD_ROOT="$OUTPUT_ROOT/shard${shard}"
  mkdir -p "$SHARD_ROOT"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$shard]}" \
  PYTHONPATH="$REPO/compat:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}" \
  OMP_NUM_THREADS=1 \
  "$PYTHON" "$REPO/scripts/run_flexhook_dance_pcd_tta.py" \
    --vendor "$VENDOR" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --tracker-root "$TRACKER_ROOT" \
    --source-result "$SOURCE_RESULT" \
    --output-root "$SHARD_ROOT" \
    --method "$METHOD" \
    --videos "${SHARD_VIDEOS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --update-interval "$UPDATE_INTERVAL" \
    > "$SHARD_ROOT/stdout.log" 2>&1 &
  PIDS[$shard]=$!
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
