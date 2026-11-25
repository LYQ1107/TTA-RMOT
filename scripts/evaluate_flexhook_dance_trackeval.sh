#!/usr/bin/env bash
set -euo pipefail

# TrackEval adapter for the released FlexHook fork.  The fork's custom
# Dance seq-length code uses this relative image path, while GT and prediction
# files are already materialized under the run's results directory.
REPO=${REPO:-/data2/user/FlexHook-TTA}
VENDOR="$REPO/vendor/FlexHook"
DATA_ROOT=${DATA_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance}
OUT=${1:-$REPO/results/runs/flexhook_v2_to_dance_yolo11x_bytetrack_compat2}

# TrackEval is launched after changing into the vendor checkout.  Normalize
# the caller-provided output path first so a relative path is not accidentally
# resolved under vendor/FlexHook.
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")

ln -sfn "$DATA_ROOT" "$VENDOR/datasets/refer-dance"
mkdir -p "$OUT"
cd "$VENDOR"
/usr/bin/python3 ./TrackEval/scripts/run_mot_challenge.py \
  --METRICS HOTA CLEAR Identity \
  --SEQMAP_FILE ./seqmaps/dance.txt \
  --SKIP_SPLIT_FOL True \
  --GT_FOLDER ./datasets/refer-kitti/KITTI/training/image_02 \
  --TRACKERS_FOLDER "$OUT/results" \
  --GT_LOC_FORMAT '{gt_folder}{video_id}/{expression_id}/gt.txt' \
  --TRACKERS_TO_EVAL "$OUT/results" \
  --USE_PARALLEL True \
  --NUM_PARALLEL_CORES 2 \
  --PLOT_CURVES False \
  > "$OUT/trackeval_repaired.log" 2>&1
