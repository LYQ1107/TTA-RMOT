#!/usr/bin/env python3
"""Prepare 64 fixed source-dev query windows; keep legal source truth off staging."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.source_kitti_v2 import SourceSupervision
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    base=Path('/data2/user/TTA-RMOT-assets/teacher-p1/calibration');stage=base/'input'
    manifest=stage/'source_manifest.jsonl';truth_file=base/'source_truth.json'
    if manifest.exists():raise FileExistsError('preserve fixed source-dev calibration manifest')
    source=SourceSupervision(ROOT/'manifests/source_kitti_v2_val.jsonl')
    plan,_=source.plan(validation_queries_per_video=16)
    if len(plan)!=64:raise ValueError('require fixed4x16 source dev query windows')
    stage.mkdir(parents=True,exist_ok=True);images=set();rows=[];truth={};seen_videos=set()
    for window in plan:
        video=window['video_id'];ep=source.episode(video);frame=window['frame'];sentence=window['query_text']
        if video not in seen_videos:
            mot=stage/'tracker_outputs'/f'{video}.txt';mot.parent.mkdir(parents=True,exist_ok=True)
            with mot.open('w') as out:
                for t,observations in ep.observations_by_frame.items():
                    for track,obs in observations.items():
                        x1,y1,x2,y2=obs.bbox_xyxy;out.write(f'{t},{track},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n')
            images.add((video,ep.frames[0]));seen_videos.add(video)
        positive=[i for i in ep.observations_by_frame[frame] if sentence in ep.membership[frame,i]]
        negative=[i for i in ep.observations_by_frame[frame] if sentence not in ep.membership[frame,i]]
        chosen=sorted(positive)[:2]+sorted(negative)[:2]
        for track in chosen:
            history=ep.history(track,frame);images.update((video,o.frame_id) for o in history)
            key=f'SOURCE/{video}/{window["expression_id"]}/{frame}/{track}'
            sample_id=hashlib.sha256(key.encode()).hexdigest()[:24]
            rows.append(dict(sample_id=sample_id,video_id=video,query_id=window['expression_id'],query_text=sentence,
                frame_id=frame,track_id=track,bbox_xyxy=list(history[-1].bbox_xyxy),history=[asdict(o) for o in history],
                attributes=[],expression_type='SOURCE_FULL_EXPRESSION',frozen_margin=0.,
                provenance='SOURCE_DEV_GT_BOXES_LEGAL_FOR_SOURCE_CALIBRATION_NOT_TARGET_MOT'))
            truth[sample_id]=dict(matches=int(sentence in ep.membership[frame,track]),video_id=video)
    for video,frame in sorted(images):
        source_path=source.root/'KITTI/training/image_02'/video/f'{frame:06d}.png'
        dest=stage/'DanceTrack/training/image_02'/video/source_path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source_path,dest)
        except OSError:shutil.copy2(source_path,dest)
    with manifest.open('w') as handle:
        for row in rows:handle.write(json.dumps(row,sort_keys=True)+'\n')
    atomic_json(truth_file,truth)
    atomic_json(base/'receipt.json',dict(protocol='LEGAL_SOURCE_DEV_CALIBRATION',windows=64,videos=4,
        candidate_queries=len(rows),manifest_sha256=file_sha(manifest),source_truth_not_in_teacher_stage=True,
        source_annotations_used=True,target_annotations_used=False,source_GT_and_images_not_published=True))
    manifest.chmod(0o444)
    for folder in sorted((p for p in stage.rglob('*') if p.is_dir()),reverse=True):folder.chmod(0o555)
    stage.chmod(0o555)
    print(json.dumps(dict(manifest=str(manifest),truth=str(truth_file),samples=len(rows)),indent=2))


if __name__=='__main__':main()
