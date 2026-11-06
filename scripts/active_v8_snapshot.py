"""Read only closed official v8 evaluation receipts, without new GT calls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_stage_snapshot import completed_evaluations


def snapshot():
    execution,_=authorize_active();public=execution.public
    plan=json.loads((public/'registered_jobs.json').read_text())
    started=json.loads((public/'execution_started.json').read_text())
    completed,failed=completed_evaluations(plan,Path(started['coordinator']))
    result=dict(status='INTERIM_V8_NOT_FINAL',completed=completed,failed=failed,
        expected=len(plan['jobs']),plan_sha256=file_sha(public/'registered_jobs.json'),
        new_GT_teacher_model_calls=0)
    atomic_json(public/'interim_official_results.json',result)
    print(json.dumps(dict(completed=len(completed),failed=len(failed),expected=len(plan['jobs']))),flush=True)


if __name__=='__main__':snapshot()
