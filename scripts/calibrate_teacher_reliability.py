#!/usr/bin/env python3
"""Freeze source-dev-only thresholds; this script cannot load Dance evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.teacher.evidence_verifier import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    truth=Path('/data2/user/TTA-RMOT-assets/teacher-p1/calibration/source_truth.json')
    labels=json.loads(truth.read_text());rows=[json.loads(s) for s in a.predictions.read_text().splitlines()]
    if {r['sample_id'] for r in rows}!=set(labels):raise ValueError('source calibration is incomplete')
    config=yaml.safe_load((ROOT/'configs/teacher/probe_protocol.yaml').read_text())['source_calibration']
    positive_grid=[];negative_grid=[];prob=[];y=[];valid=[]
    for r in rows:
        original=r['source_full_expression_views']['original'];prob.append(original['yes_probability'])
        y.append(labels[r['sample_id']]['matches']);valid.append(original['top1_is_yes_no'])
    prob,y,valid=np.asarray(prob),np.asarray(y,dtype=bool),np.asarray(valid,dtype=bool)
    for theta in config['positive_threshold_grid']:
        mask=(prob>=theta)&valid;n=int(mask.sum());precision=float(y[mask].mean()) if n else None
        positive_grid.append(dict(threshold=theta,count=n,precision=precision,
            eligible=n>=config['min_source_accepted_positive'] and precision is not None and precision>=config['require_source_precision']))
    for theta in config['negative_threshold_grid']:
        mask=(prob<=theta)&valid;n=int(mask.sum());wrong=float(y[mask].mean()) if n else None
        negative_grid.append(dict(threshold=theta,count=n,false_negative_fraction=wrong,
            eligible=n>=5 and wrong is not None and wrong<=0.05))
    pos=next((r for r in positive_grid if r['eligible']),None);neg=next((r for r in negative_grid if r['eligible']),None)
    rules=dict(positive_threshold=pos['threshold'] if pos else 0.95,negative_threshold=neg['threshold'] if neg else 0.05,
        visual_dependence_min=config['visual_dependence_min'],probability_spread_max=config['reliability_probability_spread_max'],
        status='SOURCE_RULES_FROZEN' if pos and neg else 'INSUFFICIENT_SOURCE_SUPERVISION',
        positive_calibrated=bool(pos),negative_calibrated=bool(neg),target_GT_used=False,
        source_predictions_sha256=file_sha(a.predictions),source_truth_sha256=file_sha(truth),
        source_windows=64,source_videos=4,source_candidate_queries=len(rows),source_positives=int(y.sum()),
        positive_grid=positive_grid,negative_grid=negative_grid,
        generic_source_context_warning='source full-expression calibration does not itself validate fine-grained Dance clothing semantics')
    states=[verify([dict(text='source complete description')],[r['source_full_expression_views']],
        rules['positive_threshold'],rules['negative_threshold'],rules['visual_dependence_min'],rules['probability_spread_max']).state.value for r in rows]
    accepted=np.array(states)=='SUPPORT_POS'
    rules['source_verified_POS_count']=int(accepted.sum())
    rules['source_verified_POS_precision']=float(y[accepted].mean()) if accepted.any() else None
    rules['source_verified_GT_coverage']=float(accepted[y].mean()) if y.any() else None
    rules['source_verifier_validated']=bool(accepted.sum()>=5 and y[accepted].mean()>=0.8)
    confidence=np.maximum(prob,1-prob)
    rules['softmatch_source_confidence_mean']=float(confidence.mean())
    rules['softmatch_source_confidence_variance']=float(confidence.var())
    atomic_json(a.output,rules);print(json.dumps(rules,indent=2))


if __name__=='__main__':main()
