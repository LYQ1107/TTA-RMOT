"""Resume missing fits, preserve sealed settings, then evaluate the full study.

The original sources and failure logs are immutable. Only remaining ordinary
optimization is amended; same model/data/loss/seed/capacity/adequacy gate.
"""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts import a100_joint_observation_six_setting as study

PLAN=study.PUBLIC/'joint_ordinary_optimizer_amendment_20261124.json'
STATE=study.OUT/'supervisor/ordinary_optimizer_resume_state.json'
SUCCESS='PASS_FULL_JOINT_STUDY_REPAIRED_EVALUATED_REPRODUCED_GOAL_ACTIVE'


def registration():
    original=study.check_registration()
    amendment=json.loads(PLAN.read_text())
    assert file_sha(study.PLAN)==amendment['original_registration_sha256']
    for name,sha in amendment['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    result=copy.deepcopy(original)
    result['source_sha256'].update(amendment['source_sha256'])
    result['ordinary_optimizer_amendment']=str(PLAN)
    return result


def worker(pair):
    plan=registration()
    from flex_uda.active import joint_observation_study as fit_module
    from flex_uda.active.joint_ordinary_optimizer_repair import fit_prefix
    fit_module.fit_prefix=fit_prefix
    # run() imports the fitter at invocation. Original numerical definitions,
    # checkpoints, raw path names and prediction/seal code are unchanged.
    study.check_registration=lambda:plan
    study.run(pair)


def ordinary_audit():
    rows=[]
    for pair in study.PAIRS:
        raw=json.loads((study.OUT/pair/'seed0/completion.json').read_text())
        assert raw['GT_open_attempts']==0
        for video in raw['receipts']:
            for name,fit in video['fits'].items():
                if fit['recipe']['family']=='structured':continue
                assert fit['teacher_fit_gate_passed'] and fit['metrics']['excess_KL']<=.005,(pair,video['video'],name)
                rows.append(dict(pair=pair,video=video['video'],arm=name,
                    excess_KL=fit['metrics']['excess_KL'],
                    optimizer_amendment=fit.get('optimization_amendment')))
    audit=dict(status='PASS_ALL_PREFIX_ORDINARY_FITS_BEFORE_GT',ordinary_prefix_fits=len(rows),
        maximum_excess_KL=max(r['excess_KL'] for r in rows),gate=.005,
        repaired_prefix_fits=sum(bool(r['optimizer_amendment']) for r in rows),
        GT_open_attempts=0,per_prefix=rows)
    atomic_json(study.OUT/'ordinary_fit_gate_audit_20261124.json',audit)
    return audit


def main():
    plan=registration();started=time.perf_counter()
    pending=[p for p in study.PAIRS if not(study.OUT/p/'seed0/completion.json').is_file()]
    active={};finished=[];failure=None
    def publish(status):
        view={}
        for gpu,(process,handle,job) in active.items():
            p=Path(job['folder'])/'progress.json'
            view[str(gpu)]=dict(job,progress=json.loads(p.read_text()) if p.is_file() else {})
        value=dict(status=status,pid=os.getpid(),active=view,pending=pending,finished=finished,
            failure=failure,wall_seconds=time.perf_counter()-started,neural_seeds=[0],goal_status='ACTIVE')
        atomic_json(STATE,value);print(json.dumps(value),flush=True)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    while pending or active:
        for gpu,(process,handle,job) in list(active.items()):
            code=process.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code==0 and(Path(job['folder'])/'completion.json').is_file():finished.append(job)
            else:
                log=Path(job['log']).read_text()
                failure=dict(job=job,exit_code=code,error_tail=log[-1600:])
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            raw=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in raw.strip().splitlines()}
            for gpu in candidates:
                limit=8192 if gpu in (2,3) else 512
                if not pending or memory[gpu]>=limit:continue
                pair=pending.pop(0);log=study.OUT/'supervisor'/(pair+'_optimizer_resume_20261124.log')
                handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',pair],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                job=dict(pair=pair,pid=process.pid,physical_GPU=gpu,
                    folder=str(study.OUT/pair/'seed0'),log=str(log),preflight_memory_MiB=memory[gpu])
                active[gpu]=process,handle,job
        publish('PRESERVING_HEALTHY_WORKERS_AFTER_FAILURE' if failure else 'RESUMING_MISSING_ORDINARY_FITS_NO_GT')
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(30)
    audit=ordinary_audit()
    publish('ALL_RAW_AND_ORDINARY_FITS_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(study.counts,study.PAIRS))
    tasks=[]
    for pair in study.PAIRS:
        root=study.OUT/pair/'seed0'
        for folder in (root,root/'countmatched'):
            completion=json.loads((folder/'completion.json').read_text());assert completion['GT_open_attempts']==0
            tasks.extend((str(folder/r['method']),pair) for r in completion['methods'])
    expected=6*(len(study.METHODS)+2*len(study.COMPARATORS))
    assert len(tasks)==len(set(tasks))==expected==582
    atomic_json(study.OUT/'all_predictions_sealed_before_GT.json',dict(methods=expected,
        raw_methods=6*len(study.METHODS),CountSelf_aliases=6*len(study.METHODS),GT_open_attempts=0,
        primary=study.PRIMARY,source_sha256=plan['source_sha256'],ordinary_optimizer_amendment=str(PLAN),
        all_ordinary_fit_gate_audit=str(study.OUT/'ordinary_fit_gate_audit_20261124.json')))
    publish('ALL_582_PREDICTIONS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(study.OUT/'official.json',dict(methods=expected,results=results))
    publish('ANALYZING_COMPLETE_SIX_SETTING_STUDY')
    for script in ('scripts/a100_joint_observation_analyze.py','scripts/a100_joint_observation_statistics_reproduce.py'):
        subprocess.run([sys.executable,'-u',script],cwd=ROOT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    publish(SUCCESS)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(study.PAIRS));args=parser.parse_args()
    if args.pair:worker(args.pair)
    else:main()
