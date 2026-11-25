"""Post-seal CPU analysis, forwarded into the existing blocking study pipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.a100.runtime import OUTPUT
from flex_tta.runtime.pf_receipts import atomic_json


def launch():
    for method in ('MINUS','SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY'):
        assert (OUTPUT/'stage2'/method/'predictions_seal.json').exists()
    handoff=json.loads((OUTPUT/'cache_handoff_process.json').read_text())
    destination=None
    for pid in (handoff['parent_supervisor'],handoff['pid']):
        try:destination=os.open(f'/proc/{pid}/fd/1',os.O_WRONLY);break
        except OSError:continue
    if destination is None:destination=os.open(OUTPUT/'offline_stage2_supervisor.log',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--worker'],
        cwd=ROOT,env=env,stdout=destination,stderr=subprocess.STDOUT,start_new_session=True)
    os.close(destination)
    atomic_json(OUTPUT/'stage2/offline_evaluation_process.json',dict(pid=child.pid,launched_unix=time.time(),
        only_after_all_three_prediction_seals=True,GPU_resources=0))
    print(json.dumps(dict(event='STAGE2_CPU_EVALUATION_LAUNCHED',pid=child.pid)),flush=True)


def worker():
    try:
        for phase in ('countmatched','stage2'):
            command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/a100_evaluate.py'),'--phase',phase]
            code=subprocess.call(command,cwd=ROOT)
            if code:raise RuntimeError(f'Offline phase {phase} failed with exit{code}')
        atomic_json(OUTPUT/'stage2/offline_evaluation_complete.json',dict(status='COMPLETE'))
        print(json.dumps(dict(event='STAGE2_ANALYSIS_DONE',path=str(OUTPUT/'stage2/analysis.json'))),flush=True)
    except BaseException as e:
        atomic_json(OUTPUT/'stage2/offline_evaluation_FAILED.json',dict(error=repr(e)))
        print(json.dumps(dict(event='OFFLINE_STAGE2_ERROR',status='FAILED',error=repr(e))),flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker() if a.worker else launch()
