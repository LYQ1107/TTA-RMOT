"""Schedule the unchanged registered study while adopting running workers."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data3/liuyeqiang/FlexHook-TTA');sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts import a100_observation_design_quality as study

SUP=ROOT/'a100_runs/visual_classical_20261121/supervisor'
HANDOFF=SUP/'policy_controller_adoption_input_20261123.json'


def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


class Adopted:
    def __init__(self,record):self.record=record
    def poll(self):
        if alive(self.record['pid']):return None
        return 0 if (Path(self.record['folder'])/'completion.json').is_file() else 143


def main():
    plan=json.loads(study.REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    handoff=json.loads(HANDOFF.read_text())
    while alive(handoff['old_controller_pid']):time.sleep(.2)
    start=time.perf_counter();active={};done=[];failure=None;last=0.
    for record in handoff['adopted_workers']:
        gpu=record['physical_GPU'];assert gpu not in active
        active[gpu]=Adopted(record),None,dict(record,adopted_without_restart=True)
    active_keys={(r['pair'],r['seed']) for _,_,r in active.values()}
    pending=[]
    for record in study.conditions():
        if (record['pair'],record['seed']) in active_keys:continue
        if (Path(record['folder'])/'completion.json').is_file():done.append(dict(record,reused_sealed=True))
        else:pending.append(record)
    def state(status):
        atomic_json(study.STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
            active={str(g):r for g,(_,_,r) in active.items()},pending=pending,failure=failure,
            wall_seconds=time.time()-handoff['original_registered_unix'],
            scheduler_handoff_wall_seconds=time.perf_counter()-start,
            running_models_or_worker_timers_restarted=False,goal_status='ACTIVE'))
    state('ADOPTED_EXISTING_POLICY_WORKERS_NO_GT')
    while pending or active:
        changed=False
        for gpu,(proc,handle,record) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if handle is not None:handle.close()
            del active[gpu]
            if code or not (Path(record['folder'])/'completion.json').is_file():
                log=Path(record['log']);failure=dict(job=record,exit_code=code,last_log=log.read_text()[-7000:])
            else:done.append(record)
            changed=True
        collected=study.COLLECTED/'completion.json'
        ready=collected.is_file() and json.loads(collected.read_text())['status']==study.COLLECTION_SUCCESS
        candidates=[g for g in (0,1,2,3) if g not in active]
        if ready and pending and candidates and not failure:
            text=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                record=pending.pop(0)
                if (Path(record['folder'])/'completion.json').is_file():done.append(dict(record,reused_sealed=True));changed=True;continue
                log=study.STATE.parent/(record['pair']+'_seed'+str(record['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/a100_observation_design_quality.py'),
                    '--pair',record['pair'],'--seed',str(record['seed'])],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(record,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_REGISTERED_POLICY_QUALITY_ADOPTED_WORKERS_NO_GT' if not failure else 'PRESERVING_HEALTHY_ADOPTED_POLICY_WORKERS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==18
    state('ALL18_POLICY_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(study.counts,study.conditions()))
    tasks=[]
    for record,count_root in zip(study.conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks)==828
    atomic_json(study.OUT/'all_predictions_sealed_before_GT.json',dict(methods=828,
        frozen_real_teacher_collection_complete=True,GT_open_attempts=0))
    state('ALL828_OBSERVATION_POLICY_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(study.OUT/'official.json',dict(methods=828,results=results,goal_status='ACTIVE'))
    study.analyze();state(study.SUCCESS)


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(study.STATE.read_text()) if study.STATE.is_file() else {}
        previous.update(status='FAILED_ADOPTED_POLICY_CONTROLLER',pid=os.getpid(),error=repr(error),goal_status='ACTIVE')
        atomic_json(study.STATE,previous);raise
