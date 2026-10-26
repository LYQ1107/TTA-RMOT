#!/usr/bin/env python3
"""POST-HOC diagnostic: Frozen top-k with the sealed UDA method's actual k.

Prediction construction cannot read target GT. Counts come from actual rows,
not GT, and this dependent diagnostic is not an independent deployable model.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS,protect_target_annotations
from flex_tta.runtime.pf_receipts import EpisodeStore,atomic_json
from flex_tta.dance_io import load_tracker_lines
from scripts.pf_diag_common import FROZEN,TRACKS
from scripts.diag_pf_countmatched import topk_ids


def generate(root,method):
    root=Path(root).resolve();source=root/method
    audit=json.loads((source/'full_audit.json').read_text())
    if audit['status']!='PASS' or audit['valid_episodes']!=425:raise ValueError('seal all425 predictions first')
    attempts=protect_target_annotations();name=method+'_COUNT_MATCHED_FROZEN';folder=root/name
    started=time.perf_counter();receipts=[];episodes=[];bank={}
    for entry in audit['episodes']:
        video,expression=entry['video'],entry['expression'];relative=Path('results')/video/expression
        if video not in bank:bank[video]=load_tracker_lines(TRACKS,video)
        tracks=bank[video];old=FROZEN/'ZERO_UPDATE'/relative;adapted=source/relative
        store=EpisodeStore(folder,video,expression,dict(protocol='POST_HOC_COUNT_MATCHED_DIAGNOSTIC',
            count_source=method,frozen=str(FROZEN),rule='same per-frame actual k; Frozen top-k; ID tie-break'),resume=True)
        receipt=store.completed()
        if receipt is None:
            begin=time.perf_counter();pp,mp=store.begin();mp.symlink_to((old/'candidate_margins.jsonl.gz').resolve())
            observed=[];count=selected=0
            actual_counts=Counter(int(float(x.split(',')[0])) for x in (adapted/'predict.txt').read_text().splitlines())
            with pp.open('w') as dest,gzip.open(old/'candidate_margins.jsonl.gz','rt') as src,gzip.open(adapted/'candidate_margins.jsonl.gz','rt') as other:
                for a,b in zip(src,other,strict=True):
                    a,b=json.loads(a),json.loads(b);t=a['frame'];ids=a['ids']
                    if t!=b['frame'] or ids!=b['ids'] or ids!=sorted(tracks[t]):raise ValueError('candidate universe mismatch')
                    k=actual_counts[t]
                    if k!=sum(v>0 for v in b['adapted']):raise ValueError('actual output count differs from sealed UDA score')
                    chosen=topk_ids(ids,dict(zip(ids,a['adapted'])),k)
                    for i in sorted(chosen):dest.write(tracks[t][i]+'\n')
                    observed.append(t);count+=len(ids);selected+=len(chosen)
            if observed!=sorted(tracks):raise ValueError('incomplete frames')
            receipt=store.seal(pp,mp,dict(full_frame_coverage=True,wall_seconds=time.perf_counter()-begin,
                stats=dict(frames=len(observed),candidate_rows=count,selected_rows=selected,optimizer_steps=0),
                counter_delta=dict(frames=len(observed),candidates=count,selected=selected,optimizer_steps=0)))
        receipts.append(receipt);lines=store.prediction.read_text().splitlines()
        episodes.append(dict(video=video,expression=expression,frames=receipt['stats']['frames'],
            candidates=receipt['stats']['candidate_rows'],selected_rows=len(lines),selected_ids=len({x.split(',')[1] for x in lines})))
    if len(episodes)!=425:raise ValueError('incomplete Count-Matched diagnostic')
    totals={k:sum(e[k] for e in episodes) for k in ('frames','candidates','selected_rows','selected_ids')}
    if totals['selected_rows']!=audit['selected_rows']:raise ValueError('Count-Matched totals mismatch')
    atomic_json(folder/'full_audit.json',dict(status='PASS',method=name,valid_episodes=425,unique_videos=25,
        missing_episodes=0,duplicated_episodes=0,target_gt_opened=False,tracker_geometry_changed=False,
        tracker_ids_changed=False,episodes=episodes,**totals))
    result=dict(status='SEALED',protocol='POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE',method=name,count_source=method,
        valid_episodes=425,videos=25,episodes=receipts,target_GT_open_attempts=attempts,gpu_inference_calls=0,
        optimizer_steps=0,wall_seconds=time.perf_counter()-started,
        cost_note='dependent on full UDA GPU training/inference; CPU derivation is incremental only')
    atomic_json(folder/'run_metadata.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));a=p.parse_args()
    for method in a.methods:generate(a.output_root,method)
