#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/data2/user/TTA-RMOT"
SOURCE_ROOT="/data2/user/reference_envs/temprmot/source"
SITE_ROOT="/data2/user/reference_envs/temprmot/site-packages"
RUNTIME_SITE="/data2/user/runtime_site_locatemot"
ASSET_ROOT="/data3/user/TTA-RMOT-assets/refer_kitti_v2"
TRAIN_LIST="/data2/user/reference_repos/TempRMOT/datasets/data_path/refer-kitti-v2.train"
PRETRAIN="/data3/user/TTA-RMOT-assets/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
RUN_ROOT="${TTA_RMOT_TRAIN_ROOT:-/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-7}"

mkdir -p "$RUN_ROOT"
cd "$SOURCE_ROOT" || exit 2
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="$PROJECT_ROOT/tools/temprmot_compat:$SITE_ROOT:$RUNTIME_SITE:$SOURCE_ROOT/models/ops:$SOURCE_ROOT"

if [[ -f "$RUN_ROOT/checkpoint.pth" ]]; then
    INIT_ARGS=(--resume "$RUN_ROOT/checkpoint.pth")
    echo "Resuming from $RUN_ROOT/checkpoint.pth"
else
    INIT_ARGS=(--pretrained "$PRETRAIN")
    echo "Starting from COCO Deformable-DETR initialization $PRETRAIN"
fi

CMD=(python3 -u main.py
    --meta_arch temp_rmot
    --dataset_file e2e_rmot
    --epochs 60
    --with_box_refine
    --lr 1e-4
    --lr_backbone 1e-5
    --lr_drop 40
    --batch_size 1
    --sample_mode random_interval
    --sample_interval 1
    --sampler_steps 60 80 90
    --sampler_lengths 5 5 5 5
    --update_query_pos
    --merger_dropout 0
    --dropout 0
    --random_drop 0.1
    --fp_ratio 0.3
    --query_interaction_layer QIM
    --rmot_path "$ASSET_ROOT"
    --data_txt_path_train "$TRAIN_LIST"
    --hist_len 5
    --refer_loss_coef 2
    --use_checkpoint
    --num_workers 1
    --output_dir "$RUN_ROOT"
    --save_dir "$RUN_ROOT"
    --device cuda
    "${INIT_ARGS[@]}" )

printf '%q ' "${CMD[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"
printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$RUN_ROOT/command.txt"
printf 'source_commit=%s\n' "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" >> "$RUN_ROOT/command.txt"
printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/command.txt"

"${CMD[@]}" 2>&1 | tee -a "$RUN_ROOT/train.log"
status=${PIPESTATUS[0]}
printf 'exit_status=%s\n' "$status" >> "$RUN_ROOT/command.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/command.txt"
exit "$status"
