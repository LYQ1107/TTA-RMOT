"""Publish completed official receipts while the original GPU queue continues.

No evaluator, model or GT is imported/called. Unfinished evaluations stay pending.
"""
from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def completed_evaluations(plan,coordinator):
    completed=[];failed=[]
    for job in plan['jobs']:
        path=Path(coordinator)/('evaluation_'+job['pair']+'_'+job['method']+'.json')
        if not path.exists():continue
        receipt=json.loads(path.read_text())
        if receipt['exit_code']:
            failed.append(dict(job,evaluation_receipt=str(path),exit_code=receipt['exit_code']))
            continue
        metrics=receipt['metrics']
        if metrics['GT_open_attempts']:
            raise ValueError('Prediction-generation GT isolation failed')
        seal=Path(metrics['path'])/'predictions_seal.json'
        if file_sha(seal)!=metrics['prediction_seal_sha256']:
            raise ValueError('Completed prediction seal changed')
        completed.append(dict(job,**metrics,evaluation_receipt=str(path),evaluation_receipt_sha256=file_sha(path)))
    return completed,failed


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--execution-manifest',type=Path,default=os.environ.get('PRESENCE_EXECUTION_MANIFEST'))
    parser.add_argument('--coordinator',type=Path)
    args=parser.parse_args()
    if not args.execution_manifest:parser.error('Supply the explicit execution manifest')
    execution=json.loads(Path(args.execution_manifest).read_text());public=Path(execution['public_root'])
    plan=json.loads((public/'registered_jobs.json').read_text())
    coordinator=args.coordinator
    if coordinator is None:coordinator=Path(json.loads((public/'execution_started.json').read_text())['coordinator'])
    completed,failed=completed_evaluations(plan,coordinator)
    if not completed and not failed:
        print(json.dumps(dict(status='NO_FINISHED_EVALUATIONS_NO_PUBLICATION_WRITTEN',coordinator=str(coordinator))))
        return
    result=dict(status='INTERIM_COMPLETED_OFFICIAL_EVALUATIONS_NOT_FULL_GOAL',
        completed_this_tier=len(completed),planned_this_tier=len(plan['jobs']),
        prior_completed_reused=len(plan.get('prior_completed_jobs',[])),
        original_matrix=108,remaining_after_this_full_tier=plan.get('remaining_after_this_tier'),
        coordinator=str(coordinator),plan_sha256=file_sha(public/'registered_jobs.json'),
        predictions_complete_before_official_evaluation=True,
        model_teacher_or_GT_called_by_snapshot=False,completed=completed,failed_evaluations=failed)
    atomic_json(public/'interim_official_results.json',result)
    fields=('pair','target','model','method','rho','HOTA','HOTA_unrounded','DetA','AssA','IDF1','IDSW',
        'prediction_rows','query_empty_prediction_rows','Present_HOTA','teacher_calls',
        'GT_open_attempts','future_frames','path','prediction_seal_sha256','evaluation_receipt')
    with (public/'interim_official_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(completed)
    print(json.dumps({k:v for k,v in result.items() if k not in ('completed','failed_evaluations')},indent=2))


if __name__=='__main__':main()
