"""Dispatch unchanged registered workers whose whole target evidence is ready.

No numerical helper or scientific GT barrier is changed. The global queued
controller reuses these sealed conditions after the last teacher shard ends.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data3/liuyeqiang/FlexHook-TTA');sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS
from scripts import a100_observation_design_quality as study

SUP=ROOT/'a100_runs/visual_classical_20261121/supervisor'
STATE=SUP/'ready_policy_conditions_20261123_state.json'
SUCCESS='PASS_REGISTERED_READY_POLICY_CONDITIONS_SEALED_NO_GT'


def main():
    plan=json.loads(study.REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    pending=[dict(pair=p,seed=0,folder=str(study.OUT/p/'seed0')) for p in ('C1','C3','C4')]
    for record in pending:
        for video in metadata(PAIRS[record['pair']][1])['videos']:
            for policy in study.POLICIES:
                p=study.COLLECTED/'policy_answers'/record['pair']/video/(policy+'_answers.json')
                assert p.is_file(),p
    assert json.loads(study.STATE.read_text())['status']=='AWAITING_ALL_REAL_OBSERVATION_ANSWERS_NO_GT'
    active={};finished=[];failure=None;last=0.;start=time.perf_counter()
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(finished),finished=finished,
            active={str(g):r for g,(_,_,r) in active.items()},pending=pending,failure=failure,
            wall_seconds=time.perf_counter()-start,GT_open_attempts=0,goal_status='ACTIVE'))
    state('DISPATCHING_READY_REGISTERED_POLICY_CONDITIONS')
    while pending or active:
        changed=False
        for gpu,(proc,handle,record) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code or not (Path(record['folder'])/'completion.json').is_file():
                failure=dict(record=record,exit_code=code,last_log=Path(record['log']).read_text()[-6000:])
            else:finished.append(record)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            text=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                record=pending.pop(0)
                if (Path(record['folder'])/'completion.json').is_file():finished.append(record);changed=True;continue
                log=SUP/('ready_policy_'+record['pair']+'_seed0.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/a100_observation_design_quality.py'),
                    '--pair',record['pair'],'--seed','0'],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(record,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_READY_REGISTERED_POLICY_CONDITIONS_NO_GT' if not failure else 'PRESERVING_HEALTHY_READY_POLICY_CONDITIONS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(5)
    state(SUCCESS)


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
        previous.update(status='FAILED_READY_REGISTERED_POLICY_CONDITIONS',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,previous);raise
