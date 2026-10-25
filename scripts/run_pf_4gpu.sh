#!/usr/bin/env bash
set -euo pipefail
# SAFE DEFAULT: this script prints the future four-GPU launch matrix only.
# A full run is impossible unless BOTH explicit user-approved switches are set.
REPO="${PF_REPO:-/data2/user/FlexHook-TTA}"
RUN_ROOT="${PF_RUN_ROOT:-/data2/user/FlexHook-TTA/results/runs/pf_rank_full}"
TRACKER_ROOT="${PF_TRACKER_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs}"
DATASET_ROOT="${PF_DATASET_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance}"
CHECKPOINT="${PF_CHECKPOINT:-/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth}"
SOURCE_RESULT="${PF_SOURCE_RESULT:-/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json}"
PYTHON="${PF_PYTHON:-python}"
MODE="${PF_METHOD:-PF_RANK}"
IFS=, read -r -a GPU_LIST <<< "${PF_GPUS:-0,1,2,3}"
if (( ${#GPU_LIST[@]} != 4 )); then echo "Need exactly four GPU indices" >&2; exit 2; fi
mapfile -t VIDEOS < <(find "$TRACKER_ROOT" -maxdepth 1 -type f -name 'dancetrack*.txt' -printf '%f\n' | sed 's/\.txt$//' | sort)
if (( ${#VIDEOS[@]} != 25 )); then echo "Refuse: expected 25 official Dance tracker files; got ${#VIDEOS[@]}" >&2; exit 2; fi

declare -a VIDEO_GROUPS=("" "" "" "")
for i in "${!VIDEOS[@]}"; do VIDEO_GROUPS[$((i%4))]+="${VIDEOS[$i]} "; done
for i in 0 1 2 3; do
  echo "GPU ${GPU_LIST[$i]}: ${VIDEO_GROUPS[$i]}"
done
if [[ "${1:---dry-run}" != "--launch-full" ]]; then
  echo "DRY_RUN_ONLY: no GPU process created. Full run requires explicit approval."
  exit 0
fi
if [[ "${PF_LAUNCH_FULL:-NO}" != YES || "${FLEXHOOK_FULL_RUN_AUTHORIZED:-NO}" != YES ]]; then
  echo "BLOCKED: PF_LAUNCH_FULL=YES and FLEXHOOK_FULL_RUN_AUTHORIZED=YES are both required." >&2
  exit 3
fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi required for full launch" >&2; exit 3; fi
for gpu in "${GPU_LIST[@]}"; do
  if ! nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU $gpu unavailable" >&2; exit 3
  fi
  if [[ "${PF_ALLOW_SHARED_GPU:-NO}" != YES ]] &&
     [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+' || true)" ]]; then
    echo "GPU $gpu occupied: do not interrupt other processes; choose idle GPUs" >&2; exit 3
  fi
done
PYTHONPATH="$REPO/compat:$REPO:${PYTHONPATH:-}" \
"$PYTHON" "$REPO/scripts/pf_preflight.py" --dance-root "$DATASET_ROOT" \
  --tracker-root "$TRACKER_ROOT" --checkpoint "$CHECKPOINT" --check-checkpoint-hash --strict \
  --source-result "$SOURCE_RESULT" --vendor "$REPO/vendor/FlexHook"
mkdir -p "$RUN_ROOT"
declare -a PIDS=()
for i in 0 1 2 3; do
  read -r -a SHARD_VIDEOS <<< "${VIDEO_GROUPS[$i]}"
  mkdir -p "$RUN_ROOT/shard$i"
  echo "LAUNCH GPU ${GPU_LIST[$i]} / shard$i"
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" \
  PYTHONPATH="$REPO/compat:$REPO:${PYTHONPATH:-}" \
  "$PYTHON" "$REPO/scripts/run_flexhook_positivefree.py" \
     --scope full --method "$MODE" --vendor "$REPO/vendor/FlexHook" \
     --dataset-root "$DATASET_ROOT" --tracker-root "$TRACKER_ROOT" \
     --checkpoint "$CHECKPOINT" --source-result "$SOURCE_RESULT" \
     --max-expressions 0 --max-frames 0 \
     --videos "${SHARD_VIDEOS[@]}" --output-root "$RUN_ROOT/shard$i" \
     >"$RUN_ROOT/shard$i/stdout.log" 2>&1 &
  PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if (( status )); then echo "One or more full shards failed" >&2; exit "$status"; fi
echo "ALL SHARDS COMPLETE. Merge and evaluate only after output audits."
