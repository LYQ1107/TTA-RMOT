#!/usr/bin/env python3
"""Replay all sealed Frozen scores without GT, then evaluate in another process."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl
from flex_uda.data.target_dance import TargetDanceDataset
from flex_uda.training import protect_target_annotations
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import EpisodeStore,atomic_json,file_sha

SOURCE=ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE'


def replay(output):
    rows=TargetDanceDataset(ROOT/'manifests/dance_val_runtime.jsonl',split='val').rows
    attempts=protect_target_annotations();started=time.perf_counter();receipts=[];episodes=[];banks={}
    folder=Path(output)/'FROZEN'
    for row in rows:
        video,expression=row['video_id'],row['expression_id']
        old=SOURCE/'results'/video/expression
        tracks=banks.setdefault(video,None)
        if tracks is None:
            tracks=load_tracker_lines(Path(row['predicted_tracks']).parent,video);banks[video]=tracks
        store=EpisodeStore(folder,video,expression,dict(protocol='SEALED_FROZEN_REPLAY',
            source_root=str(SOURCE),threshold=0,candidate_coverage='COMPLETE_211085'),resume=True)
        receipt=store.completed()
        if receipt is None:
            pp,mp=store.begin();mp.symlink_to((old/'candidate_margins.jsonl.gz').resolve())
            seen=[];candidate_count=selected_count=0;begin=time.perf_counter()
            with pp.open('w') as dest,gzip.open(old/'candidate_margins.jsonl.gz','rt') as stream:
                for line in stream:
                    record=json.loads(line);t=record['frame'];ids=record['ids'];scores=record['adapted']
                    if ids!=sorted(tracks[t]) or len(ids)!=len(scores):raise ValueError('Frozen candidate mismatch')
                    chosen={i for i,s in zip(ids,scores) if s>0}
                    for i in sorted(chosen):dest.write(tracks[t][i]+'\n')
                    seen.append(t);candidate_count+=len(ids);selected_count+=len(chosen)
            if seen!=sorted(tracks):raise ValueError('Frozen frame coverage mismatch')
            if file_sha(pp)!=file_sha(old/'predict.txt'):raise ValueError('Frozen raw prediction differs from sealed source')
            receipt=store.seal(pp,mp,dict(full_frame_coverage=True,wall_seconds=time.perf_counter()-begin,
                stats=dict(frames=len(seen),candidate_rows=candidate_count,selected_rows=selected_count,optimizer_steps=0),
                counter_delta=dict(frames=len(seen),candidates=candidate_count,selected=selected_count,optimizer_steps=0)))
        receipts.append(receipt)
        episodes.append(dict(video=video,expression=expression,frames=receipt['stats']['frames'],
            candidates=receipt['stats']['candidate_rows'],selected_rows=receipt['stats']['selected_rows'],
            selected_ids=len({x.split(',')[1] for x in store.prediction.read_text().splitlines()})))
    if len(receipts)!=425 or len(banks)!=25:raise ValueError('Frozen must cover25videos/425queries')
    atomic_json(folder/'full_audit.json',dict(status='PASS',method='FROZEN',valid_episodes=425,unique_videos=25,
        missing_episodes=0,duplicated_episodes=0,target_gt_opened=False,tracker_geometry_changed=False,
        tracker_ids_changed=False,episodes=episodes,**{k:sum(e[k] for e in episodes) for k in ('frames','candidates','selected_rows','selected_ids')}))
    result=dict(status='PASS_425_EXACT_PREDICTION_HASHES',protocol='SEALED_FROZEN_REPLAY',valid_episodes=425,videos=25,
        threshold=0,gpu_inference_calls=0,optimizer_steps=0,source=str(SOURCE),episodes=receipts,
        target_GT_open_attempts=attempts,wall_seconds=time.perf_counter()-started,
        cost_note='CPU replay of historical GPU predictions; not a newly measured GPU baseline')
    atomic_json(folder/'run_metadata.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();replay(a.output_root.resolve())
