#!/usr/bin/env python3
"""Publish completed confirmation training, without touching healthy inference."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def snapshot():
    public=ROOT/'results/psr/confirmations';public.mkdir(parents=True,exist_ok=True)
    jobs=[(f'seed{s}',m,ROOT/f'results/runs/psr_confirm_20261030_v1/seed{s}')
        for s in (0,42) for m in ('NAIVE_PAIRED','PSR')]
    jobs.append(('parent_weight','PSR_PARENT_WEIGHT_CONTROL',ROOT/'results/runs/psr_parent_weight_20261030_v1'))
    states={}
    for name,method,run in jobs:
        folder=run/'students'/method;receipt=folder/'training_receipt.json';key=name+'/'+method
        if not receipt.exists():states[key]='NOT_YET_COMPLETE';continue
        r=json.loads(receipt.read_text())
        if r['status']!='COMPLETE_FIXED_COVERAGE_PSR_STUDY' or r['steps']!=8973 or r['trainable_parameters']!=24576 or r['source_videos_seen']!=17 or r['target_query_pairs_seen']!=425 or r['target_GT_open_attempts'] or not r['frozen_parameters_unchanged']:raise ValueError('invalid completed training: '+key)
        dest=public/name/'training'/method;dest.mkdir(parents=True,exist_ok=True)
        for filename in ('training_receipt.json','provenance.json','config.json','source_validation.jsonl','teacher_fit_summary.json','teacher_fit_after_training.jsonl.gz'):
            if (folder/filename).exists():shutil.copy2(folder/filename,dest/filename)
        with (folder/'metrics_train.jsonl').open('rb') as src,(dest/'metrics_train.jsonl.gz').open('wb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
        source=run/'students'/(method+'_stdout.txt')
        if source.exists():
            with source.open('rb') as src,(dest/'training_stdout.txt.gz').open('wb') as dst:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
        for source in (folder/'attempt_runtimes').glob('*.json'):
            target=dest/'attempt_runtimes'/source.name;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
        states[key]='COMPLETE_TRAINING_ONLY'
    record=dict(status='INTERMEDIATE_COMPLETED_TRAINING_NOT_FULL_EVALUATION',methods=states,
        protocol='USER_AUTHORIZED25VAL_SAME_SET_NOT_TRAIN_TO_VAL',GT_reads=0,GPU_calls=0,
        does_not_authorize_early_target_GT_evaluation=True,does_not_imply_scientific_success=True,
        inference_and_official_results='pending original blocking queue; do not restart healthy units')
    atomic_json(public/'training_stage_history'/f'{time.time_ns()}.json',record)
    print(json.dumps(record))


if __name__=='__main__':snapshot()
