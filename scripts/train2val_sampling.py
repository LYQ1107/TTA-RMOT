#!/usr/bin/env python3
"""Reuse the exact Naive8500 sampler on Train; independent Val replay test."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl,write_jsonl
from flex_uda.data.train2val import ASSETS
from flex_uda.teacher.psr_sampler import sample_episode
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def scores_for(path):
    with gzip.open(path,'rt') as stream:
        return {(r['frame'],i):s for r in map(json.loads,stream) for i,s in zip(r['ids'],r['adapted'])}


def prepare():
    output=ASSETS/'pseudo';folder=output/'NAIVE';folder.mkdir(parents=True,exist_ok=True)
    receipt=output/'manifest_receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if file_sha(folder/'manifest.jsonl')!=old['manifests_sha256']['NAIVE']:raise ValueError('changed manifest')
        return old
    runtime=read_jsonl(ASSETS/'runtime_train.jsonl');groups=defaultdict(list)
    for r in runtime:groups[r['video_id']].append(r)
    samples=[];windows=[];selection=[];hashes={};image_hashes={};stage=output/'inputs'
    for video,queries in sorted(groups.items()):
        first=queries[0];images=Path(first['image_root']);dataset=images.parents[3]
        ep=DanceEpisode(dataset,ASSETS/'tracker_outputs',video)
        observations={t:{i:dict(bbox_xyxy=list(o.bbox_xyxy),raw_mot_line=o.raw_line) for i,o in v.items()} for t,v in ep.observations_by_frame.items()}
        target=stage/'tracker_outputs'/(video+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(first['predicted_tracks'],target)
        hashes[str(Path(first['predicted_tracks']).resolve())]=file_sha(first['predicted_tracks'])
        used={ep.frames[0]}
        for row in queries:
            scores=scores_for(ASSETS/'frozen_sparse/results'/video/row['expression_id']/'candidate_margins.jsonl.gz')
            ss,ww,rr=sample_episode(row,observations,scores,'NAIVE',seed=17)
            samples.extend(ss);windows.extend(ww);selection.extend(rr)
            for s in ss:used.update((s['anchor_frame']-1,s['anchor_frame'],s['anchor_frame']+1))
        for frame in sorted(used):
            src=images/f'{frame:06d}.png';dst=stage/'DanceTrack/training/image_02'/video/src.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():os.link(src,dst)
            image_hashes[f'{video}/{frame}']=file_sha(src)
    for name,rows in [('runtime.jsonl',runtime),('manifest.jsonl',samples),('windows.jsonl',windows)]:
        write_jsonl(folder/name,rows);(folder/name).chmod(0o444)
    atomic_json(folder/'selection_audit.json',selection)
    result=dict(status='FROZEN_MATCHED_BUDGET_PSR_MANIFESTS',target_split='train',protocol='FORMAL_TRAIN_TO_VAL',
        runtime_videos=len(groups),runtime_queries=len(runtime),seed=17,paired_single_frame_teacher=True,
        samples_per_policy={'NAIVE':len(samples)},windows=len(windows),actual_per_query_budgets_match=True,
        manifests_sha256={'NAIVE':file_sha(folder/'manifest.jsonl')},tracker_files_sha256=hashes,
        image_files_sha256=image_hashes,teacher_input_staging=str(stage),target_GT_open_attempts=[],unasked_are_NOT_NEG=True,
        candidate_source='ORIGINAL_DANCETRACK_TRAIN_GT_FRAME_ID_BOX_ONLY')
    atomic_json(receipt,result);return result


def replay():
    """Separate audit process only; never called from any training worker."""
    original=Path('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE')
    samples=read_jsonl(original/'manifest.jsonl');runtime=read_jsonl(original/'runtime.jsonl')
    by=defaultdict(list)
    for s in samples:by[s['video_id'],s['query_id']].append(s)
    frozen=ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE'
    episodes={};checked=0
    for row in runtime:
        v,q=row['video_id'],row['expression_id']
        if v not in episodes:
            ep=DanceEpisode(Path(row['image_root']).parents[3],Path(row['predicted_tracks']).parent,v)
            episodes[v]={t:{i:dict(bbox_xyxy=list(o.bbox_xyxy),raw_mot_line=o.raw_line) for i,o in vals.items()} for t,vals in ep.observations_by_frame.items()}
        scores=scores_for(frozen/'results'/v/q/'candidate_margins.jsonl.gz')
        generated,_,_=sample_episode(row,episodes[v],scores,'NAIVE',seed=17)
        expected=by[v,q]
        keys=('sample_id','video_id','query_id','frame_id','track_id','bbox_xyxy','anchor_frame','window_id','selection_reasons','budget_role','frozen_margin')
        if [{k:s[k] for k in keys} for s in generated]!=[{k:s[k] for k in keys} for s in expected]:
            raise RuntimeError('original Naive8500 sampling replay mismatch: '+v+'/'+q)
        checked+=len(generated)
    result=dict(status='PASS_EXACT_NAIVE8500_SAMPLING_REPLAY',queries=len(runtime),samples=checked,
        manifest_sha256=file_sha(original/'manifest.jsonl'),referring_GT_opened=False,
        separate_audit_process_only_not_a_training_input=True,compared_fields=list(keys))
    atomic_json(ROOT/'results/train2val/naive_sampling_replay.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('train','replay'));a=p.parse_args()
    result=prepare() if a.phase=='train' else replay()
    print(json.dumps({k:v for k,v in result.items() if k not in ('tracker_files_sha256','image_files_sha256')},indent=2))
