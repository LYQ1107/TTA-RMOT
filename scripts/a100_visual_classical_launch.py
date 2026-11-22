"""Two independent domain workers; one blocking supervisor, no foreign kills."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
OUT=ROOT/'a100_runs/visual_classical_20261121'


def main():
    folder=OUT/'supervisor';folder.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter();children=[]
    for pair,gpu in [('C4',0),('C1',1)]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
        log=(folder/(pair+'.log')).open('a')
        # The worker itself runs preparation followed by collection, so CUDA
        # startup is not paid until its unlabelled visual inputs are ready.
        command=[str(ROOT/'.venv/bin/python'),str(__file__),'--worker',pair]
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((pair,gpu,child,log))
    atomic_json(folder/'started.json',dict(jobs=[dict(pair=p,gpu=g,pid=c.pid) for p,g,c,_ in children],
        no_foreign_process_stopped=True))
    receipts=[]
    for pair,gpu,child,log in children:
        code=child.wait();log.close();receipts.append(dict(pair=pair,gpu=gpu,exit_code=code))
    result=dict(status='PASS' if all(r['exit_code']==0 for r in receipts) else 'FAILED',jobs=receipts,
        wall_seconds=time.perf_counter()-begin)
    atomic_json(folder/'completion.json',result);print(json.dumps(result),flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__':
    if len(sys.argv)>1:
        from scripts.a100_visual_classical_prepare import prepare
        from scripts.a100_visual_classical_collect import collect
        pair=sys.argv[2];prepare(pair);collect(pair)
    else:sys.exit(main())
