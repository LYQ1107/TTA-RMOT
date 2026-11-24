"""Independent six-setting quality test of the distribution-space RSI update.

Reuses the pinned native binary worker and numerical inputs. Only this
process binds a reduced method scope and the prospective actor update.
The original 726-evaluation study remains unchanged and running.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts import a100_native_binary_six_setting as original

PAIRS=original.PAIRS
OUT=original.OUT.parent/'native_binary_trust_quality_20261124'
PUBLIC=original.PUBLIC
PLAN=PUBLIC/'native_binary_trust_quality_registration_20261124.json'
STATE=OUT/'supervisor/state.json'
PRIMARY=original.PRIMARY
RECIPES={n:r for n,r in original.RECIPES.items() if n in (
    'FULL_CONDITIONAL_REFERENCE_LEARNED','FULL_CONDITIONAL_REFERENCE_FROZEN')
    or (r['family']!='structured' and r['policy']=='LEARNED')}
METHODS=tuple(n+'_'+s for n,r in RECIPES.items()
              for s in (('FINAL',) if r['family']=='structured' else ('FINAL','MEAN')))
COMPARATORS=tuple(n for n in METHODS if n!=PRIMARY)
EXPECTED=6*(len(METHODS)+2*len(COMPARATORS))


def check_plan():
    plan=json.loads(PLAN.read_text())
    assert plan['neural_seeds']==[0] and plan['recipes']==RECIPES and plan['methods']==list(METHODS)
    assert plan['primary']==PRIMARY and plan['official_unique_evaluations']==EXPECTED==312
    for n,s in plan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s,n
    parent=json.loads(original.PLAN.read_text())
    assert hashlib.sha256(original.PLAN.read_bytes()).hexdigest()==plan['parent_registration_sha256']
    for n,s in parent['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s,n
    return plan


def bind_scope():
    original.OUT=OUT;original.PRIMARY=PRIMARY;original.RECIPES=RECIPES
    original.METHODS=METHODS;original.COMPARATORS=COMPARATORS
    original.registration=check_plan


def worker(pair):
    check_plan();bind_scope()
    from flex_uda.active import native_binary_curriculum as curriculum
    from flex_uda.active.rsi_policy_distribution import recorded_selection,gated_update,trust_update
    credit_function=curriculum.progress_reward;credit={}
    def measure(*args):
        credit.clear();credit.update(credit_function(*args));return dict(credit)
    def update(*args,**kwargs):return gated_update(trust_update,credit,*args,**kwargs)
    curriculum.select_round=recorded_selection;curriculum.progress_reward=measure;curriculum.update_challenger=update
    original.worker(pair)


def counts(pair):
    bind_scope();original.counts(pair)


def main():
    plan=check_plan();pending=list(PAIRS);active={};finished=[];failure=None;start=time.perf_counter()
    def state(status):
        view={}
        for gpu,(proc,handle,job) in active.items():
            p=Path(job['folder'])/'progress.json'
            view[str(gpu)]=dict(job,progress=json.loads(p.read_text()) if p.is_file() else {})
        row=dict(status=status,pid=os.getpid(),active=view,pending=pending,finished=finished,failure=failure,
            wall_seconds=time.perf_counter()-start,neural_seeds=[0],goal_status='ACTIVE')
        atomic_json(STATE,row);print(json.dumps(row),flush=True)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    while pending or active:
        for gpu,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code==0 and(Path(job['folder'])/'completion.json').is_file():finished.append(job)
            else:failure=dict(job=job,exit_code=code,error_tail=Path(job['log']).read_text()[-2200:])
        for pair in list(pending):
            if(OUT/pair/'seed0/completion.json').is_file():pending.remove(pair);finished.append(dict(pair=pair,reused_sealed=True))
        candidates=[g for g in (2,3,0,1) if g not in active]
        if pending and candidates and not failure:
            raw=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=(8192 if gpu in (2,3) else 512):continue
                if gpu in (0,1) and subprocess.check_output(['nvidia-smi','-i',str(gpu),
                    '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():continue
                pair=('C1' if 'C1' in pending else 'C6' if 'C6' in pending else pending[0]);pending.remove(pair)
                log=OUT/'supervisor'/(pair+'_seed0.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',pair],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                job=dict(pair=pair,pid=proc.pid,physical_GPU=gpu,folder=str(OUT/pair/'seed0'),log=str(log),preflight_memory_MiB=memory[gpu])
                active[gpu]=proc,handle,job
        state('PRESERVING_HEALTHY_TRUST_WORKERS_AFTER_FAILURE' if failure else 'RUNNING_RSI_TRUST_SIX_SETTINGS_NO_GT')
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(30)
    state('ALL_TRUST_RAW_SEALED_AUDITING_ORDINARY_AND_GENERATING_COUNTS_NO_GT')
    fits=[]
    for pair in PAIRS:
        raw=json.loads((OUT/pair/'seed0/completion.json').read_text());assert raw['GT_open_attempts']==0
        for v in raw['receipts']:
            for name,report in v['fits'].items():
                if RECIPES[name]['family']=='structured':continue
                for f in report['fits']:
                    assert f['teacher_fit_gate_passed'] and f['excess_risk']<=.005
                    fits.append(dict(pair=pair,video=v['video'],arm=name,excess_risk=f['excess_risk'],units=f['risk_units']))
    atomic_json(OUT/'all_ordinary_fit_gate_before_GT.json',dict(fits=fits,GT_open_attempts=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(counts,PAIRS))
    tasks=[]
    for pair in PAIRS:
        root=OUT/pair/'seed0'
        for folder in (root,root/'countmatched'):
            complete=json.loads((folder/'completion.json').read_text());assert complete['GT_open_attempts']==0
            tasks.extend((str(folder/r['method']),pair) for r in complete['methods'])
    assert len(tasks)==len(set(tasks))==EXPECTED
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=EXPECTED,raw_methods=6*len(METHODS),
        CountSelf_aliases=6*len(METHODS),GT_open_attempts=0,primary=PRIMARY,source_sha256=plan['source_sha256'],
        cohort='native_binary_trust_quality_20261124'))
    state('TRUST_ALL_OUTPUTS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=EXPECTED,results=results))
    state('ANALYZING_COMPLETE_RSI_TRUST_SIX_SETTINGS')
    for script in ('scripts/a100_native_binary_trust_analyze.py','scripts/a100_native_binary_trust_statistics_reproduce.py'):
        subprocess.run([sys.executable,'-u',script],cwd=ROOT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    state('PASS_RSI_TRUST_SIX_SETTING_QUALITY_EXECUTION_REPRODUCED_GOAL_ACTIVE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(PAIRS));args=parser.parse_args()
    if args.pair:worker(args.pair)
    else:main()
