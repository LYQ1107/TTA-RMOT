#!/usr/bin/env bash
set -euo pipefail

# Full Refer-Dance validation with the released Refer-KITTI-V2 second-stage
# checkpoint.  Tracker files are generated separately and must not contain
# ground-truth boxes or referring expressions.
REPO=${REPO:-/data2/user/FlexHook-TTA}
VENDOR="$REPO/vendor/FlexHook"
DATA_ROOT=${DATA_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance}
TRACK_ROOT=${TRACK_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/runs/v2_to_dance_yolo11x_bytetrack/tracker_outputs}
CHECKPOINT=${CHECKPOINT:-/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth}
OUT=${OUT:-$REPO/results/runs/flexhook_v2_to_dance_yolo11x_bytetrack_compat2}
PYTHON=${PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}
PORT=${PORT:-12349}
GT_LINK="$VENDOR/datasets/refer-kitti/KITTI/training/image_02"

ORIGINAL_GT=$(readlink "$GT_LINK")
test -n "$ORIGINAL_GT"
mkdir -p "$OUT"

restore_gt() {
  ln -sfn "$ORIGINAL_GT" "$GT_LINK"
}
trap restore_gt EXIT
ln -sfn "$DATA_ROOT/gt_template" "$GT_LINK"

cd "$VENDOR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,5}" \
PYTHONPATH="$REPO/compat:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}" \
OMP_NUM_THREADS=1 \
"$PYTHON" -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
  --master-port "$PORT" main.py \
  --cfg configs/infer/dance.yaml \
  --track-root "$TRACK_ROOT" \
  --output "$OUT" \
  --val-batch-size 80 \
  --visual rope-swin-tiny \
  --text roberta \
  --eval \
  --resume "$CHECKPOINT" \
  --opts data_root "$DATA_ROOT"
