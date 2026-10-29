#!/usr/bin/env python3
"""Freeze matched-budget paired manifests from runtime text/MOT/Frozen margins."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl,write_jsonl,official_splits
from flex_uda.teacher.psr_sampler import MODES,sample_episode
from flex_uda.training import protect_target_annotations
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def prepare(split,output,frozen_root,val_authorized=False):
    if split=='val' and not val_authorized:raise PermissionError('explicit user Val fallback required')
    output=Path(output).resolve();frozen_root=Path(frozen_root).resolve();attempts=protect_target_annotations()
    runtime=read_jsonl(ROOT/'manifests'/('dance_val_runtime.jsonl' if split=='val' else 'dance_train_uda.jsonl'))
    if {r['video_id'] for r in runtime}!=set(official_splits()['dance'][split]):raise ValueError('incomplete official runtime split')
    for r in runtime:
        if not Path(r['predicted_tracks']).is_file():raise FileNotFoundError('missing fixed MOT; never substitute Val or GT: '+r['predicted_tracks'])
    receipt=output/'manifest_receipt.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['target_split']!=split:raise ValueError('different previous protocol')
        for mode,sha in saved['manifests_sha256'].items():
            if file_sha(output/mode/'manifest.jsonl')!=sha:raise ValueError('frozen manifest changed')
        print(json.dumps(dict(status='PRESERVED_FROZEN_MANIFESTS',samples=saved['samples_per_policy'])));return saved
    output.mkdir(parents=True,exist_ok=True);stage=output/'inputs';groups=defaultdict(list)
    for row in runtime:groups[row['video_id']].append(row)
    samples={m:[] for m in MODES};windows={m:[] for m in MODES};selections={m:[] for m in MODES};hashes={};image_hashes={}
    for video,queries in sorted(groups.items()):
        first=queries[0];image_root=Path(first['image_root']);dataset=image_root.parents[3]
        ep=DanceEpisode(dataset,Path(first['predicted_tracks']).parent,video)
        observations={t:{i:dict(bbox_xyxy=list(o.bbox_xyxy),raw_mot_line=o.raw_line) for i,o in values.items()} for t,values in ep.observations_by_frame.items()}
        tracks=stage/'tracker_outputs'/f'{video}.txt';tracks.parent.mkdir(parents=True,exist_ok=True)
        if not tracks.exists():shutil.copy2(first['predicted_tracks'],tracks)
        hashes[str(Path(first['predicted_tracks']).resolve())]=file_sha(first['predicted_tracks'])
        used={ep.frames[0]}
        for runtime_row in sorted(queries,key=lambda r:r['expression_id']):
            start=runtime_row['frame_indices']['start'];last=runtime_row['frame_indices']['stop']-1
            anchors={start+round((last-start)*q) for q in (.1,.3,.5,.7,.9)}
            needed={t+d for t in anchors for d in (-1,0,1)};scores={}
            source=frozen_root/'results'/video/runtime_row['expression_id']/'candidate_margins.jsonl.gz'
            if not source.is_file():raise FileNotFoundError('Frozen source score bank missing for current split: '+str(source))
            with gzip.open(source,'rt') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['frame'] in needed:
                        if row['ids']!=sorted(observations.get(row['frame'],{})):raise ValueError('fixed candidate/source score alignment differs')
                        scores.update({(row['frame'],i):s for i,s in zip(row['ids'],row['adapted'])})
            budgets=[]
            for mode in MODES:
                ss,ww,rr=sample_episode(runtime_row,observations,scores,mode)
                samples[mode].extend(ss);windows[mode].extend(ww);selections[mode].extend(rr);budgets.append(len(ss))
                for sample in ss:used.update((sample['anchor_frame']-1,sample['anchor_frame'],sample['anchor_frame']+1))
            if len(set(budgets))!=1:raise ValueError('unmatched actual per-query teacher budget')
        for frame in sorted(used):
            source=image_root/f'{frame:06d}.png';target=stage/'DanceTrack/training/image_02'/video/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(source,target)
            image_hashes[f'{video}/{frame}']=file_sha(source)
        print(json.dumps(dict(event='psr_sampling_video',video=video,samples_each={m:len(v) for m,v in samples.items()})),flush=True)
    for mode in MODES:
        folder=output/mode;folder.mkdir(exist_ok=True)
        write_jsonl(folder/'manifest.jsonl',samples[mode]);write_jsonl(folder/'windows.jsonl',windows[mode]);write_jsonl(folder/'runtime.jsonl',runtime)
        atomic_json(folder/'selection_audit.json',selections[mode])
        for name in ('manifest.jsonl','windows.jsonl','runtime.jsonl'):(folder/name).chmod(0o444)
    result=dict(status='FROZEN_MATCHED_BUDGET_PSR_MANIFESTS',target_split=split,
        protocol='VAL_SAME_SET_DIAGNOSTIC_NOT_TRAIN_TO_VAL' if split=='val' else 'FORMAL_TRAIN_TO_VAL',
        source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime_videos=len(groups),runtime_queries=len(runtime),seed=17,paired_single_frame_teacher=True,
        samples_per_policy={m:len(v) for m,v in samples.items()},actual_per_query_budgets_match=True,
        manifests_sha256={m:file_sha(output/m/'manifest.jsonl') for m in MODES},tracker_files_sha256=hashes,
        image_files_sha256=image_hashes,teacher_input_staging=str(stage),target_GT_open_attempts=attempts,
        unasked_are_NOT_NEG=True)
    atomic_json(receipt,result);print(json.dumps({k:v for k,v in result.items() if k not in ('tracker_files_sha256','image_files_sha256')},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=('train','val'),default='train');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen-root',type=Path,required=True);p.add_argument('--val-authorized',action='store_true');a=p.parse_args();prepare(a.split,a.output,a.frozen_root,a.val_authorized)
