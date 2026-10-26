#!/usr/bin/env python3
"""Actual UDA asset/source-split audit; never reads target GT payloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import SOURCE,official_splits,read_jsonl
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

CHECKPOINT=Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth')
EXPECTED='71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613'


def run():
    actual=file_sha(CHECKPOINT)
    if actual!=EXPECTED:raise ValueError('V2 checkpoint differs from verified source model')
    split=official_splits();source_manifest=Path('/data2/user/reference_repos/TempRMOT/datasets/data_path/refer-kitti-v2.train')
    source_videos=sorted({Path(line).parent.name for line in source_manifest.read_text().splitlines() if line.strip()})
    if len(source_videos)!=17 or set(source_videos)&set(split['kitti-2']['val']):raise ValueError('source complete-train split differs')
    # Source annotations are explicitly permitted; target labels are NOT opened.
    source_labels=json.loads((SOURCE/'labels.json').read_text())
    videos=[]
    for video in sorted(set(source_videos)|set(split['kitti-2']['val'])):
        image_root=SOURCE/'KITTI/training/image_02'/video
        frames=sorted(int(p.stem) for p in image_root.glob('*.png'))
        expression_files=sorted((SOURCE/'expression'/video).glob('*.json'))
        if not frames or not expression_files or video not in source_labels:raise ValueError('source data incomplete: '+video)
        videos.append(dict(video_id=video,split='train' if video in source_videos else 'val',frames=len(frames),
            images_contiguous=frames==list(range(frames[0],frames[-1]+1)),expressions=len(expression_files),
            annotated_track_ids=len(source_labels[video]),image_root=str(image_root)))
    source_rows=[dict(video_id=v['video_id'],source_root=str(SOURCE),source_labels_path=str(SOURCE/'labels.json'),
        image_root=v['image_root'],expression_root=str(SOURCE/'expression'/v['video_id'])) for v in videos]
    from flex_uda.data.manifests import write_jsonl
    write_jsonl(ROOT/'manifests/source_kitti_v2_train.jsonl',[r for r in source_rows if r['video_id'] in source_videos])
    write_jsonl(ROOT/'manifests/source_kitti_v2_val.jsonl',[r for r in source_rows if r['video_id'] not in source_videos])
    target=json.loads((ROOT/'results/uda/assets/dance_manifest_audit.json').read_text())
    release=json.loads((ROOT/'results/uda/assets/official_tracker_release.json').read_text())
    commands={}
    for name,cmd in [('gpus',['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        ('gpu_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
        ('disk',['df','-h']),('inodes',['df','-i']),('mounts',['findmnt'])]:
        commands[name]=subprocess.check_output(cmd,text=True)
    result=dict(status='ASSET_BLOCKED_NO_FORMAL_UDA_STARTED' if not target['ready_for_formal_training'] else 'ASSETS_READY',
        observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),project='LYQ1107/TTA-RMOT',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        checkpoint=dict(path=str(CHECKPOINT),sha256=actual,verified=True),
        source=dict(complete_train_manifest=str(source_manifest),complete_train_manifest_sha256=file_sha(source_manifest),
            complete_train_video_ids=source_videos,complete_train_videos=17,source_val_videos=4,
            vendored_flexhook_default_train_ids=split['kitti-2']['train'],
            explicit_full_source_difference=sorted(set(source_videos)-set(split['kitti-2']['train'])),
            train_frames=sum(v['frames'] for v in videos if v['split']=='train'),
            train_expressions=sum(v['expressions'] for v in videos if v['split']=='train'),videos=videos),
        target=dict(train_videos=40,val_videos=25,train_expressions=target['train_episodes'],val_expressions=target['val_episodes'],
            train_frames=sum(v['images'] for v in target['inventory'] if v['split']=='train'),train_val_overlap=[],
            missing_train_tracker_videos=[r['video_id'] for r in target['inventory'] if r['split']=='train' and not r['predicted_tracks_present']],
            public_release_status=release['status'],public_release_train_files=len(release.get('released_train_video_ids',[])),
            public_release_val_files=len(release.get('released_val_video_ids',[]))),
        target_gt_payloads_read=False,source_gt_access_authorized=True,gpu_and_filesystem_snapshot=commands,
        gpu_snapshot_is_reservation=False,formal_training_started=False,
        next_required_asset='40 official Train DETR+NeuralSORT outputs OR the exact pretrained front-end weights/config/code; no GT-derived replacement')
    atomic_json(ROOT/'results/uda/assets/asset_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source','gpu_and_filesystem_snapshot')},indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result['source'].items() if k!='videos'},indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':run()
