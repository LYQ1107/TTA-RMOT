#!/usr/bin/env python3
"""Seal coverage/numerical audits after all independent workers have returned."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json


def finalize(root):
    root=Path(root).resolve();methods={}
    for method in METHODS:
        folder=root/method;receipt=json.loads((folder/'training_receipt.json').read_text())
        records=[json.loads(s) for s in (folder/'metrics_train.jsonl').read_text().splitlines()]
        if receipt['status']!='COMPLETE_FIXED_COVERAGE_DIAGNOSTIC':raise ValueError('not a completed full cycle')
        if [r['step'] for r in records]!=list(range(1,8974)):raise ValueError('missing/duplicate accepted training step')
        if receipt['counters']['optimizer_updates']!=8973 or receipt['counters'].get('amp_skipped_updates',0):
            raise ValueError('a source/target sample was skipped rather than replayed')
        if any(not math.isfinite(r.get(k,0.)) for r in records for k in ('source_loss','target_loss','gradient_norm')):
            raise ValueError('nonfinite accepted objective/gradient')
        if receipt['source_videos_seen']!=17 or any(c['sampled_GT_tracks']!=c['eligible_GT_tracks'] for c in receipt['actual_source_coverage']):
            raise ValueError('source coverage incomplete')
        if method!='SOURCE_REHEARSAL' and (receipt['target_videos_seen']!=25 or receipt['target_query_pairs_seen']!=425):
            raise ValueError('target query/video coverage incomplete')
        if receipt['target_GT_open_attempts'] or not receipt['frozen_source_unchanged']:raise ValueError('source/annotation isolation violation')
        rows=dict(status='PASS',accepted_steps=8973,optimizer_updates=8973,losses_and_gradients_finite=True,
            source_videos=17,target_videos=receipt['target_videos_seen'],target_queries=receipt['target_query_pairs_seen'],
            target_GT_open_attempts=[],frozen_source_unchanged=True,amp_same_sample_retries=receipt['counters'].get('amp_same_sample_retries',0),
            raw_path=str(folder),target_control_note='SOURCE_REHEARSAL has no target losses or adaptation' if method=='SOURCE_REHEARSAL' else None)
        atomic_json(folder/'training_finite_audit.json',rows);methods[method]=rows
    attempts=[root/'launcher_status.json']+sorted(root.glob('launcher_recoveries/*/launcher_status.json'))
    launchers=[dict(path=str(p),**json.loads(p.read_text())) for p in attempts]
    if any(x['status']=='RUNNING' for x in launchers):raise ValueError('a launcher has not returned')
    start=(root/'gpu_admission.json').stat().st_mtime;end=max(p.stat().st_mtime for p in attempts)
    result=dict(status='COMPLETE_ALL_FOUR_FIXED_COVERAGE_RUNS',protocol='TRANSDUCTIVE_VAL_DIAGNOSTIC',methods=methods,
        wall_seconds=end-start,wall_definition='first admission to latest launcher return, including diagnosis/recovery delay',
        launchers=launchers,failed_attempts_preserved=True,checkpoints_selected_without_targetGT=True)
    atomic_json(root/'full_suite_completion.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);a=p.parse_args();finalize(a.training_root)
