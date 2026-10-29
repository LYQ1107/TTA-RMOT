#!/usr/bin/env python3
"""Separate true/false teacher-positive transfer using sealed offline CSV only.

Training-anchor fit is NOT independent validation and never selects a model,
sampling strategy, threshold, or loss weight.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.analyze_psr_matrix import soft_fit_by_state


def analyze(public):
    public=Path(public).resolve();answer={}
    # These CSVs were created only after complete prediction sealing and GT
    # attribution. This script never opens raw target GT or launches a model.
    for policy in ('NAIVE','PSR','UNIFORM','CONFIDENCE'):
        with (public/(policy+'_direct_teacher_offline.csv')).open() as stream:records=list(csv.DictReader(stream))
        methods=[key.removeprefix('student_margin_') for key in records[0] if key.startswith('student_margin_')]
        for row in records:
            row['official_match']=row['official_match']=='True'
            for key in ('p','source_margin',*('student_margin_'+m for m in methods)):row[key]=float(row[key])
        groups={}
        for name,correct in [('qualified_POS_official_match',True),('qualified_POS_official_unmatched',False)]:
            subset=[r for r in records if r['state']=='POS' and r['official_match']==correct]
            fits=soft_fit_by_state(subset,methods)['methods'];source_selected=sum(r['source_margin']>0 for r in subset);students={}
            for method in methods:
                key='student_margin_'+method
                students[method]=dict(selected=sum(r[key]>0 for r in subset),
                    newly_selected=sum(r[key]>0 and r['source_margin']<=0 for r in subset),
                    lost_source_selected=sum(r[key]<=0 and r['source_margin']>0 for r in subset),
                    source_selected_retained=sum(r[key]>0 and r['source_margin']>0 for r in subset),
                    fit=fits[method]['ALL'])
            groups[name]=dict(candidate_rows=len(subset),source_selected=source_selected,
                source_fit=fits['FROZEN']['ALL'],students=students)
        answer[policy]=groups
    result=dict(status='COMPLETE_FROM_SEALED_OFFLINE_CANDIDATE_CSV',policies=answer,GPU_calls=0,raw_target_GT_files_opened=False,
        checkpoint_scope='source-val-selected checkpoint on its own teacher training anchors',
        not_independent_validation=True,not_used_for_checkpoint_threshold_or_loss_selection=True,
        identity_scope='candidate observations, not distinct people or one-to-one GT coverage; see teacher_positive_support.json for unique GT coverage',
        GT_scope='official IoU>=.5 annotation match, not exhaustive real-world semantic truth')
    atomic_json(public/'correct_positive_fit.json',result)
    print(json.dumps({p:{'correct_POS':r['qualified_POS_official_match']['candidate_rows'],
        'source_selected':r['qualified_POS_official_match']['source_selected'],
        'student_selected':{m:s['selected'] for m,s in r['qualified_POS_official_match']['students'].items()}} for p,r in answer.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-root',type=Path,default=ROOT/'results/psr/full')
    analyze(p.parse_args().public_root)
