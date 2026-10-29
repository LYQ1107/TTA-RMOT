#!/usr/bin/env python3
"""Audit formal Train inputs and exact frontend availability, without target GT."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl,official_splits,TRAIN_TRACKS,ASSETS
from flex_uda.training import protect_target_annotations,CHECKPOINT,SOURCE_SHA
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def audit(output,refresh=False):
    attempts=protect_target_annotations();rows=read_jsonl(ROOT/'manifests/dance_train_uda.jsonl')
    split=official_splits()['dance'];expected=set(split['train']);videos={r['video_id'] for r in rows}
    if videos!=expected or expected&set(split['val']):raise ValueError('official split changed')
    records=[]
    for video in sorted(expected):
        row=next(r for r in rows if r['video_id']==video);images=Path(row['image_root'])
        frames={int(p.stem) for p in images.glob('*.png')};path=TRAIN_TRACKS/(video+'.txt')
        record=dict(video=video,images=len(frames),queries=sum(r['video_id']==video for r in rows),MOT_path=str(path),present=path.is_file())
        if path.is_file():
            keys=set();track_frames={}
            for line in path.read_text().splitlines():
                values=list(map(float,line.split(',')))
                if len(values)!=10 or not all(map(math.isfinite,values)):raise ValueError('invalid MOT format: '+video)
                frame,track=int(values[0]),int(values[1])
                if values[0]!=frame or values[1]!=track or frame not in frames or min(values[4:6])<=0 or (frame,track) in keys:raise ValueError('invalid MOT frame/ID/bbox: '+video)
                keys.add((frame,track));track_frames.setdefault(track,[]).append(frame)
            if not keys:raise ValueError('empty MOT file: '+video)
            record.update(rows=len(keys),sha256=file_sha(path),frame_offset=0,
                discontinuous_predicted_IDs=sum(any(b-a>1 for a,b in zip(sorted(fs),sorted(fs)[1:])) for fs in track_frames.values()),
                ID_note='predicted IDs may drift/reappear; this audit does not claim physical ground-truth identity')
        records.append(record)
    references={}
    for name,path in [('FlexHook',ROOT/'vendor/FlexHook'),('iKUN',ASSETS.parent/'mllm-distill/references/iKUN')]:
        references[name]=dict(path=str(path),python_sources=[str(p.relative_to(path)) for p in path.rglob('*.py') if 'TrackEval' not in p.parts],
            conclusion='released referring model consumes precomputed MOT; no exact Dance DETR+NeuralSORT frontend inference/config/checkpoint was found in these sources')
    archive=ASSETS/'Refer-Dance.zip';prediction_members=[]
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if any(token in name.lower() for token in ('detr','neuralsort','tracker_output','detection')):prediction_members.append(name)
    receipt=TRAIN_TRACKS.parent/'frontend_provenance.json'
    provenance=json.loads(receipt.read_text()) if receipt.exists() else None
    missing=[r['video'] for r in records if not r['present']]
    ready=not missing and provenance is not None and provenance.get('frontend')=='OFFICIAL_FLEXHOOK_DETR_NEURALSORT'
    result=dict(status='FORMAL_TRAIN_READY' if ready else 'MISSING_EXACT_FORMAL_TRAIN_FRONTEND_ASSETS',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        source_checkpoint_sha256=file_sha(CHECKPOINT),official_source_SHA_matches=file_sha(CHECKPOINT)==SOURCE_SHA,
        train_videos=40,train_queries=len(rows),train_images=sum(r['images'] for r in records),train=records,
        missing_train_videos=missing,frontend_provenance=provenance,formal_training_ready=ready,
        dataset_ZIP_prediction_members=prediction_members,ZIP_metadata_only_no_GT_payload_read=True,references=references,
        target_GT_open_attempts=attempts,allow_Val_substitution=False,
        recovery_conditions=['author-released40Train DETR+NeuralSORT output files with source/provenance, OR exact official detector and NKF tracker code/config/weights sufficient to reproduce the fixedVal frontend',
            'audit all40 predicted files against raw images, seal hashes, then run registered source scoring/teacher generation/training'],
        no_alternate_detector_or_GT_tracks_used=True)
    if refresh:
        from scripts.audit_uda_tracker_release import inspect
        try:result['fresh_official_public_release']=inspect()
        except Exception as error:result['fresh_official_public_release']=dict(status='REQUEST_FAILED',error_type=type(error).__name__,message=str(error))
    atomic_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','references','fresh_official_public_release')},ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/psr/assets/formal_train_audit.json');p.add_argument('--refresh-release',action='store_true');a=p.parse_args();audit(a.output,a.refresh_release)
