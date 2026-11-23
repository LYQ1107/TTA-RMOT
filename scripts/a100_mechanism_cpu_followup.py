"""Blocking CPU supervisor: registered controls, official evaluation, no polling agent."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import OUT,OLD


def run(script,args,name):
    directory=OUT/'cpu_supervisor';directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/(name+'.json')
    if receipt.is_file() and json.loads(receipt.read_text())['status']=='COMPLETE':return
    attempt=len(list(directory.glob(name+'_attempt*.log')));logpath=directory/f'{name}_attempt{attempt:02d}.log'
    command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script),*args]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT))
    begin=time.perf_counter();print(json.dumps(dict(event='CPU_START',job=name)),flush=True)
    with logpath.open('w') as log: code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    row=dict(status='COMPLETE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.perf_counter()-begin,command=command,log=str(logpath))
    atomic_json(receipt,row);print(json.dumps(dict(event='CPU_END',job=name,**row)),flush=True)
    if code:raise RuntimeError('CPU child failed: '+str(logpath))


def evaluate(folder):
    from scripts.a100_mechanism_offline import evaluate as official
    official(folder,'C4')


def main():
    assert json.loads((OUT/'fixed_replay/C4/reference_equivalence.json').read_text())['status']=='PASS'
    original=OLD/'stage2/MINUS';replay=OUT/'fixed_replay/C4/DS__EXPLICIT_TASK'
    for filename in ('pedestrian_detailed.csv','pedestrian_summary.txt'):
        shutil.copy2(original/'results'/filename,replay/'results'/filename)
    result=json.loads((original/'official_completion.json').read_text())
    result.update(reused_after_all_466_prediction_hashes_equal=True,official_reference=str(original),
        prediction_seal_sha256=file_sha(replay/'predictions_seal.json'),additional_evaluation_wall_seconds=0.)
    atomic_json(replay/'official_completion.json',result)
    def temporal_lane():
        run('a100_mechanism_replay.py',['--phase','temporal','--pair','C4'],'temporal_generate')
        run('a100_mechanism_evaluate_controls.py',['--group','temporal'],'temporal_evaluate')
    def protocol_lane():
        run('a100_mechanism_replay.py',['--phase','replay','--pair','C4','--fusion','DS','--protocol','UNIFIED_CHECKLIST'],'protocol_unified_generate')
        run('a100_mechanism_evaluate_controls.py',['--group','protocol'],'protocol_unified_evaluate')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,'a100_mechanism_offline.py',['--pair','C4','--phase','oracles'],'oracle_evaluate'),
                 pool.submit(temporal_lane),pool.submit(protocol_lane)]
        for f in concurrent.futures.as_completed(futures):f.result()
    # Evidence collection has its own blocking supervisor on GPU0. Waiting here
    # is internal, so the root agent never repeatedly inspects PIDs/log tails.
    while not (OUT/'evidence/completion.json').is_file():
        p=OUT/'evidence/supervisor_receipt.json'
        if p.is_file() and json.loads(p.read_text())['status']=='FAILED':raise RuntimeError('Evidence producer failed')
        time.sleep(5)
    run('a100_mechanism_bank.py',[],'bank_decisions_generate')
    def bank_arm(fusion,routing):
        name=fusion+'__EXPLICIT_TASK__BANK_'+routing
        run('a100_mechanism_replay.py',['--phase','replay','--pair','C4','--fusion',fusion,
            '--protocol','EXPLICIT_TASK','--bank-routing',routing],name+'_generate')
        run('a100_mechanism_evaluate_controls.py',['--group','bank','--name',name],name+'_evaluate')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(bank_arm,f,r) for f in ('DS','FIXED') for r in ('ORIGINAL','CROP')]
        for future in concurrent.futures.as_completed(futures):future.result()
    atomic_json(OUT/'cpu_supervisor/completion.json',dict(status='ALL_REGISTERED_CPU_CONTROLS_COMPLETE'))


if __name__=='__main__':main()
