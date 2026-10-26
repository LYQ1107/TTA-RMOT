#!/usr/bin/env python3
"""Offline-only pseudo-label quality AFTER all four425 predictions are sealed."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.analyze_pf_full_offline import load_eval_gt,overlaps
from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes
from scripts.pf_diag_common import DATA,TRACKS,write_csv


def analyze(training,evaluation,public):
    training,evaluation,public=map(lambda p:Path(p).resolve(),(training,evaluation,public))
    for method in METHODS:
        a=json.loads((evaluation/method/'full_audit.json').read_text())
        if a['status']!='PASS' or a['valid_episodes']!=425:raise ValueError('seal every method before any pseudo GT review')
    boxes={};gt={};reports={};table=[]
    for method in ('MEAN_TEACHER_UDA','COMBINED_UDA'):
        counts=Counter();unique=set();unique_counts=Counter();groups={k:Counter() for k in ('Query-Present','Query-Empty')}
        with (training/method/'pseudo_evidence.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);video,q,t,i=row['video_id'],row['query_id'],row['frame_id'],row['track_id'];state=row['pseudo_state']
                if video not in boxes:boxes[video]=load_tracker_boxes(TRACKS,video)
                if (video,q) not in gt:gt[video,q]=load_eval_gt(DATA,video,q)
                labels=gt[video,q];present=bool(labels);key=(video,q,t,i,state)
                matrix=overlaps([boxes[video][t][i]],[box for _,box in labels.get(t,[])])
                positive=bool((matrix>=.5).any());label='GT_MATCH' if positive else 'NO_GT_MATCH'
                name=state+'_'+label;counts[name]+=1;groups['Query-Present' if present else 'Query-Empty'][name]+=1
                if key not in unique:unique_counts[name]+=1;unique.add(key)
        pos=counts['POS_GT_MATCH']+counts['POS_NO_GT_MATCH'];neg=counts['NEG_GT_MATCH']+counts['NEG_NO_GT_MATCH']
        result=dict(exposures=dict(counts),unique_candidate_state_events=dict(unique_counts),groups={k:dict(v) for k,v in groups.items()},
            pseudo_POS_precision=counts['POS_GT_MATCH']/pos if pos else None,
            pseudo_NEG_correct_fraction=counts['NEG_NO_GT_MATCH']/neg if neg else None,
            unknown_GT_positive_exposures=counts['UNKNOWN_GT_MATCH'],
            unknown_is_not_negative=True,unique_event_key='video,query,frame,track,pseudo_state; state may change through training',
            precision_definition='IoU>=0.5 to an official scored GT for this query; not one-to-one tracking precision',
            correlation_warning='repeated training exposures are not independent samples; no frame-level confidence intervals',
            selection_warning='quality of all historical training evidence, not GT-based selection of the final checkpoint')
        reports[method]=result
        for state in ('POS','NEG','UNKNOWN'):
            table.append(dict(method=method,pseudo_state=state,GT_match_exposures=counts[state+'_GT_MATCH'],
                no_GT_match_exposures=counts[state+'_NO_GT_MATCH'],unique_GT_match_events=unique_counts[state+'_GT_MATCH'],
                unique_no_GT_match_events=unique_counts[state+'_NO_GT_MATCH']))
    result=dict(protocol='OFFLINE_GT_DIAGNOSTIC_ONLY',target_GT_read_after_all_four425_predictions_sealed=True,
        target_GT_used_in_training=False,target_GT_used_for_threshold_or_checkpoint=False,methods=reports)
    atomic_json(public/'pseudo_label_quality_offline.json',result);write_csv(public.parent/'pseudo_label_quality_offline.csv',table)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--evaluation-root',type=Path,required=True);p.add_argument('--public-root',type=Path,default=ROOT/'results/uda/full')
    a=p.parse_args();analyze(a.training_root,a.evaluation_root,a.public_root)
