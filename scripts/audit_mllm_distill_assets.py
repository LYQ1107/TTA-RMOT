#!/usr/bin/env python3
"""Audit fixed inputs and reusable baselines without opening target GT."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
from flex_uda.training import CHECKPOINT,SOURCE_SHA,protect_target_annotations
from flex_uda.data.manifests import read_jsonl,write_jsonl,TRAIN_TRACKS
from flex_uda.data.source_kitti_v2 import SourceSupervision
from flex_uda.teacher.mllm_pseudo_dataset import MLLMPseudoDataset
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def audit():
    attempts=protect_target_annotations();public=ROOT/'results/mllm_distillation/assets';public.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((ROOT/'configs/uda/mllm_naive_distill.yaml').read_text())
    pseudo=MLLMPseudoDataset(config['data']['pseudo_directory'],allow_val_diagnostic=True)
    release=json.loads(Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/official_detr_neuralsort_receipt.json').read_text())
    official={r['file']:r['sha256'] for r in release['files']};tracks=[]
    for video in sorted({r['video_id'] for r in pseudo.runtime}):
        row=next(r for r in pseudo.runtime if r['video_id']==video);path=Path(row['predicted_tracks']);sha=file_sha(path)
        if sha!=official[path.name]:raise ValueError('official Val MOT hash mismatch')
        frames={int(p.stem) for p in Path(row['image_root']).glob('*.png')};seen=set();track_ids=set();active=set()
        for line in path.read_text().splitlines():
            f=list(map(float,line.split(',')))
            if len(f)!=10 or not all(math.isfinite(v) for v in f):raise ValueError('malformed fixed MOT')
            t,i=int(f[0]),int(f[1])
            if f[0]!=t or f[1]!=i or t not in frames or f[4]<=0 or f[5]<=0 or (t,i) in seen:raise ValueError('invalid frame/ID/bbox/duplicate')
            seen.add((t,i));track_ids.add(i);active.add(t)
        tracks.append(dict(video=video,sha256=sha,images=len(frames),MOT_rows=len(seen),MOT_active_frames=len(active),
            images_without_MOT=len(frames-active),predicted_track_ids=len(track_ids),frame_offset=0,MOT_frames_one_based=True))
    source=SourceSupervision(ROOT/config['data']['source_train_manifest']);plan,coverage=source.plan()
    plan_path=ROOT/'results/runs/mllm_val_distill_20261028_v1/asset_audit_source_plan.jsonl';write_jsonl(plan_path,plan)
    old=ROOT/'results/uda/full/SOURCE_REHEARSAL';old_config=json.loads((old/'config.json').read_text())
    old_train=json.loads((old/'training_receipt.json').read_text());old_prov=json.loads((old/'provenance.json').read_text())
    old_checkpoint=json.loads((old/'checkpoint_metadata.json').read_text())
    old_trace=[json.loads(s) for s in gzip.open(old/'metrics_train.jsonl.gz','rt')]
    checks=dict(source_init_SHA_matches=file_sha(CHECKPOINT)==SOURCE_SHA,
        source_plan_SHA_matches=file_sha(plan_path)==old_prov['plan_sha256']['source_train_plan.jsonl'],
        optimizer=old_config['optimizer']==config['optimizer'],adapter=old_config['adapter']==config['adapter'],
        seed=old_config['protocol']['seed']==config['protocol']['seed'],
        train_budget=old_config['train']==config['train'] and old_train['steps']==len(plan),
        selection=old_config['selection']==config['selection'],source_loss=old_config['loss']['source_ce']==config['loss']['source_ce']==1.,
        no_target_optimization=old_train['target_query_pairs_seen']==0 and all(r['target_rows']==0 for r in old_trace),
        all_optimizer_updates=len(old_trace)==old_train['counters']['optimizer_updates']==8973,
        frozen_and_GT_audit=old_train['frozen_source_unchanged'] and not old_train['target_GT_open_attempts'],
        checkpoint_SHA=file_sha(Path(old_checkpoint['checkpoint']))==old_checkpoint['checkpoint_sha256'])
    reuse=all(checks.values())
    baseline=dict(source_only_reuse_eligible=reuse,checks=checks,source_train_windows=len(plan),source_videos=len(source.rows),
        source_coverage=coverage,source_only_checkpoint=old_checkpoint,source_only_existing_results=str(old),
        note='SOURCE_REHEARSAL performs no target optimization. Same source code/plan/seed/initialization/optimizer/capacity/selection/steps; retain original result and cost, no relabelled retraining.',
        mean_teacher_use='HISTORICAL_REFERENCE_NOT_MATCHED_MLLM_OR_TARGET_SAMPLING_BUDGET',
        mean_teacher_differences=['EMA teacher, not external MLLM','3 target anchors per query vs current5','all MOT candidates vs current<=4 sampled candidates','different target losses and supervision/call budget'],
        Frozen_complete_bank_baseline_HOTA=8.2180,Frozen_protocol='complete211085officialMOT observations, sourceV2, FP32 last-layer margin>0,425expressions',
        target_GT_open_attempts=attempts)
    atomic_json(public/'baseline_protocol_audit.json',baseline)
    train=read_jsonl(ROOT/'manifests/dance_train_uda.jsonl')
    result=dict(project='LYQ1107/TTA-RMOT',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        protocol=config['protocol'],user_authorized_Val_instead_of_missing_Train_MOT=True,source_checkpoint_sha256=SOURCE_SHA,
        Val_videos=len(tracks),Val_expressions=len(pseudo.runtime),Val_MOT_rows=sum(r['MOT_rows'] for r in tracks),Val_track_audit=tracks,
        Train_videos=len({r['video_id'] for r in train}),Train_expressions=len(train),
        Train_MOT_directory=str(TRAIN_TRACKS),Train_MOT_present=TRAIN_TRACKS.is_dir(),
        missing_Train_MOT_videos=sorted({r['video_id'] for r in train if not Path(r['predicted_tracks']).is_file()}),
        teacher_seal_path=str(pseudo.directory/'seal.json'),teacher_seal_sha256=file_sha(pseudo.directory/'seal.json'),
        source_GT_legal=True,target_GT_open_attempts=attempts,frontend_reproduction='author-released Val output hashes verified; exact detector/NKF checkpoints not supplied by FlexHook code; no invented Train predictions',
        official_references=['https://github.com/buptLwz/FlexHook','https://github.com/dyhBUPT/iKUN'],
        source_only_reuse_eligible=reuse)
    atomic_json(public/'data_model_audit.json',result)
    print(json.dumps(dict(Val_videos=result['Val_videos'],Val_expressions=result['Val_expressions'],Val_MOT_rows=result['Val_MOT_rows'],
        Train_MOT_present=result['Train_MOT_present'],source_only_reuse_eligible=reuse,checks=checks,target_GT_open_attempts=attempts),indent=2))
    return result


if __name__=='__main__':audit()
