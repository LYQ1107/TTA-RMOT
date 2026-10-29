#!/usr/bin/env python3
"""GT-free descriptive summaries of completed, real student optimization."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json


def summarize(folder):
    folder=Path(folder);receipt=json.loads((folder/'training_receipt.json').read_text())
    if (receipt['status']!='COMPLETE_FIXED_COVERAGE_PSR_STUDY' or receipt['trainable_parameters']!=24576 or
        receipt['counters']['optimizer_updates']!=8973 or receipt['source_videos_seen']!=17 or receipt['target_query_pairs_seen']!=425 or
        receipt['target_GT_open_attempts'] or not receipt['frozen_parameters_unchanged'] or not receipt['checkpoint_roundtrip']):
        raise ValueError('completed student numerical/data/coverage acceptance failed: '+str(folder))
    if receipt['method']=='MEAN_TEACHER_MATCHED':
        if receipt['counters']['EMA_updates']!=8973 or receipt['external_MLLM_labels_used'] or receipt['unique_unlabelled_target_inputs_seen']!=8500:raise ValueError('matchedEMA coverage/state invalid')
    elif receipt['unique_teacher_samples_seen']!=8500:raise ValueError('incomplete direct8500 actual coverage')
    rows=[json.loads(s) for s in (folder/'metrics_train.jsonl').read_text().splitlines()]
    if receipt['steps']!=8973 or [r['step'] for r in rows]!=list(range(1,8974)):raise ValueError('full8973 chronological trace required')
    fields=('source_loss','target_KD_loss','expanded_KD_loss','expanded_weighted_loss','source_gradient_norm',
        'direct_gradient_norm','expanded_gradient_norm','gradient_norm','source_direct_gradient_cosine',
        'source_expanded_gradient_cosine','direct_expanded_gradient_cosine')
    groups={}
    for name,chosen in [('all_steps',rows),('with_qualified_direct_POS',[r for r in rows if r['direct_states'].get('POS',0)]),
            ('without_qualified_direct_POS',[r for r in rows if not r['direct_states'].get('POS',0)]),
            ('with_expanded_weak',[r for r in rows if r['expanded_valid_rows']])]:
        result=dict(steps=len(chosen),fraction_of_8973=len(chosen)/8973,statistics={})
        for field in fields:
            values=[r[field] for r in chosen if r.get(field) is not None]
            if any(not math.isfinite(v) for v in values):raise ValueError('nonfinite actual dynamics: '+field)
            result['statistics'][field]=dict(defined_steps=len(values),mean=float(np.mean(values)) if values else None,
                p10_p50_p90=np.percentile(values,[10,50,90]).tolist() if values else None)
        groups[name]=result
    states={kind:Counter() for kind in ('direct','expanded')}
    for r in rows:
        for kind in states:states[kind].update(r[kind+'_states'])
    return dict(method=receipt['method'],optimizer_updates=receipt['counters']['optimizer_updates'],
        groups=groups,repeated_exposure_states={k:dict(v) for k,v in states.items()},
        unique_direct_teacher_labels=receipt['unique_teacher_samples_seen'],unique_weak_labels=receipt['unique_expanded_samples_seen'],
        external_target_semantic_teacher=receipt['method']!='MEAN_TEACHER_MATCHED',
        actual_training_acceptance=dict(source_videos=17,target_query_pairs=425,trainable_parameters=24576,frozen_unchanged=True,checkpoint_roundtrip=True),
        target_objective='0.2EMA consistency +0.2confidence pseudo BCE, not external MLLM KD' if receipt['method']=='MEAN_TEACHER_MATCHED' else 'raw frozen MLLM soft probability BCE',
        target_GT_reads=0,gradient_norm_note='per-step preclip module/combined norms; not norm of the average gradient or causal contribution',
        gradient_weight_note='expanded_gradient_norm already includes fixed0.25 weak coefficient; expanded_KD_loss is raw and expanded_weighted_loss is weighted',
        POS_definition='frozen MLLM confidence qualification, or internal EMA pseudo state for matchedMT; not GT correctness',
        scientific_scope='fitting/optimization evidence only, NOT RMOT generalization or independent semantic recovery')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={m:summarize(a.training_root/m) for m in a.methods};atomic_json(a.output,dict(status='COMPLETE_FOR_LISTED_FINISHED_METHODS',target_GT_reads=0,methods=result))
    print(json.dumps(dict(methods=list(result),status='GT_FREE_TRAINING_DYNAMICS_COMPLETE')))
