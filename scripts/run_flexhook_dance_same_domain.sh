#!/usr/bin/env bash
set -euo pipefail

# Full same-domain ceiling run: the released Dance second-stage checkpoint
# evaluated on the released Dance DETR-NeuralSORT candidate bank.
REPO=${REPO:-/data2/user/FlexHook-TTA}
VENDOR="$REPO/vendor/FlexHook"
DATA_ROOT=${DATA_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance}
TRACK_ROOT=${TRACK_ROOT:-/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs}
CHECKPOINT=${CHECKPOINT:-/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-dance-best.pth}
OUT=${OUT:-$REPO/results/runs/flexhook_dance_to_dance_official_detr_neuralsort}
PYTHON=${PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}
PORT=${PORT:-12350}
NPROC=${NPROC:-1}
GT_LINK="$VENDOR/datasets/refer-kitti/KITTI/training/image_02"

test -f "$CHECKPOINT"
test -d "$DATA_ROOT"
test -d "$TRACK_ROOT"
ORIGINAL_GT=$(readlink "$GT_LINK")
test -n "$ORIGINAL_GT"
mkdir -p "$OUT"

restore_gt() {
  ln -sfn "$ORIGINAL_GT" "$GT_LINK"
}
trap restore_gt EXIT
ln -sfn "$DATA_ROOT/gt_template" "$GT_LINK"

cd "$VENDOR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" \
PYTHONPATH="$REPO/compat:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}" \
OMP_NUM_THREADS=1 \
"$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC" --nnodes=1 \
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
