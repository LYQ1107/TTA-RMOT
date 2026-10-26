#!/usr/bin/env python3
"""Seal exact25/425 evaluations and account for staggered one-GPU launches."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json


def finalize(root):
    root=Path(root).resolve();methods={}
    for method in METHODS:
        folder=root/method;a=json.loads((folder/'full_audit.json').read_text())
        if a['status']!='PASS' or a['valid_episodes']!=425 or a['unique_videos']!=25:raise ValueError('incomplete25/425 prediction tree')
        if a['target_gt_opened'] or a['tracker_geometry_changed'] or a['tracker_ids_changed']:raise ValueError('prediction protocol violation')
        metadata=json.loads((folder/'run_metadata.json').read_text())
        if metadata['target_GT_open_attempts'] or metadata['test_time_updates']:raise ValueError('invalid evaluation isolation')
        methods[method]=dict(valid_episodes=425,frames=a['frames'],candidate_query_rows=a['candidates'],selected_rows=a['selected_rows'])
    paths=sorted(root.glob('evaluation_launchers/*/evaluation_launcher_status.json'));launchers=[];intervals=[]
    if not paths:raise ValueError('missing actual evaluator launch records')
    for path in paths:
        row=json.loads(path.read_text())
        if row['status']!='COMPLETE':raise ValueError('an evaluator launch failed or is unfinished')
        start=(path.parent/'evaluation_gpu_admission.json').stat().st_mtime
        intervals.append((start,start+row['wall_seconds']));launchers.append(dict(path=str(path),**row))
    intervals.sort();merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    elapsed=max(end for start,end in intervals)-min(start for start,end in intervals)
    result=dict(status='COMPLETE_ALL_FOUR_25VIDEO_425QUERY_EVALUATIONS',methods=methods,wall_seconds=elapsed,
        union_evaluator_active_wall_seconds=sum(b-a for a,b in merged),launchers=launchers,
        first_evaluator_admission_unix=min(a for a,b in intervals),last_evaluator_return_unix=max(b for a,b in intervals),
        wall_definition='first inference launch to last return; may overlap training of slower methods; never add blindly to training wall')
    atomic_json(root/'evaluation_suite_completion.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();finalize(a.output_root)
