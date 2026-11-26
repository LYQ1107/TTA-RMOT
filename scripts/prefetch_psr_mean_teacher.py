#!/usr/bin/env python3
"""Start the registered EMA comparator after Discovery releases its GPU lane.

One CPU blocking wait. Existing queued copies deduplicate via the validated
per-output CLI locks, before any CUDA allocation. No healthy job is stopped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def prefetch(run):
    run=Path(run).resolve();source=run/'evaluation/merged/PSR_DISCOVERY';start=time.perf_counter()
    print(json.dumps(dict(event='WAIT_FOR_DISCOVERY_GPU_RELEASE',registered_method='MEAN_TEACHER_MATCHED',
        target_GT_reads=0,waiting_process_uses_GPU=False)),flush=True)
    while True:
        audit=source/'full_audit.json'
        ended=[json.loads(p.read_text()) for p in (source/'inference_attempts').glob('*.json')]
        if audit.exists() and any(r['exit_code']==0 for r in ended):
            status=json.loads(audit.read_text())
            if status['status']=='PASS' and status['valid_episodes']==425 and status['unique_videos']==25:break
        time.sleep(20)
    # Other original lanes remain registered on 1/5/4. This launch may use
    # at most one further physical GPU and re-audits actual free memory/counts.
    command=[sys.executable,str(ROOT/'scripts/run_psr_matrix.py'),'--root',str(run),
        '--methods','MEAN_TEACHER_MATCHED','--workers','1','--other-active-gpus','1','5','4']
    print(json.dumps(dict(event='START_RELEASED_LANE_MEAN_TEACHER',command=command,CPU_wait_seconds=time.perf_counter()-start)),flush=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    child=subprocess.Popen(command,cwd=ROOT,env=env);code=child.wait()
    atomic_json(run/'mean_teacher_prefetch_completion.json',dict(status='COMPLETE' if code==0 else 'FAILED_RETAINED',
        exit_code=code,seconds=time.perf_counter()-start,method='MEAN_TEACHER_MATCHED',
        preserved_peer_GPUs=[1,5,4],maximum_combined_physical_GPUs=4,source_inference_completed=True,
        duplicate_queued_work='per-output CLI locks/preserved full receipts; no repeated optimizer or episode prediction'))
    if code:raise RuntimeError('inspect this failed launch only; original healthy queue is unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();prefetch(a.run_root)
