"""Publish only finished official receipts while the original v7 queue runs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_stage_snapshot import completed_evaluations
from scripts.report_active import write_table


def snapshot():
    execution,_=authorize_active();public=execution.public
    plan=json.loads((public/'registered_jobs.json').read_text())
    started=json.loads((public/'execution_started.json').read_text());coordinator=Path(started['coordinator'])
    completed,failed=completed_evaluations(plan,coordinator)
    result=dict(status='INTERIM_V7_OFFICIAL_RECEIPTS_NOT_FINAL',completed_jobs=len(completed),
        expected_primary_jobs=len(plan['jobs']),completed=completed,failed_evaluations=failed,
        plan_sha256=file_sha(public/'registered_jobs.json'),coordinator=str(coordinator),
        new_evaluator_model_teacher_or_GT_calls=0,v6_extension_or_remaining80_started=False)
    atomic_json(public/'interim_official_results.json',result)
    write_table(public/'interim_official_results.csv',completed)
    print(json.dumps({k:v for k,v in result.items() if k not in ('completed','failed_evaluations')},indent=2))


if __name__=='__main__':snapshot()
