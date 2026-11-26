"""Stage1 pilots: first two C1/C2 videos, resumable, no target GT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    execution,_=authorize_active()
    assert execution.manifest['version']=='v8'
    assert json.loads((execution.public/'amendments_A2_A3/cpu_tests.json').read_text())['failed']==0
    plan=json.loads((execution.public/'registered_jobs.json').read_text())
    plan['jobs']=[j for j in plan['jobs'] if j['pair'] in ('C1','C2') and j['method']=='MINUS_SLOW']
    pilot_plan=execution.public/'pilot_jobs.json';atomic_json(pilot_plan,plan)
    subprocess.run([sys.executable,str(ROOT/'scripts/active_v8_launch.py'),'--phase','pilot',
        '--pilot-videos','2','--jobs-json',str(pilot_plan)],cwd=ROOT,check=True)
    records=[]
    for job in plan['jobs']:
        root=execution.active/job['pair']/'MINUS_SLOW/rho_0.3'
        receipt=json.loads((root/'pilot_receipt.json').read_text())
        assert receipt['GT_open_attempts']==0 and len(receipt['videos'])==2
        for video in receipt['videos']:
            for r in range(5):assert (root/'videos'/video/f'presence_round{r}.npz').exists()
        assert receipt['em']['converged']
        records.append(dict(receipt,path=str(root),pilot_receipt_sha256=file_sha(root/'pilot_receipt.json')))
    atomic_json(execution.public/'gpu_smoke.json',dict(status='V8_TWO_DIRECTION_TWO_VIDEO_PILOTS_PASS',
        records=records,GT_open_attempts=0,no_RMOT_GT_evaluated=True,
        full_teacher_cost_in_later_resumed_stream=True,maximum_physical_GPUs=2))
    print(json.dumps(dict(event='V8_STAGE1_COMPLETE',pilots=2,videos=4,GT_open_attempts=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs-json',type=Path);p.parse_args()
    main()
