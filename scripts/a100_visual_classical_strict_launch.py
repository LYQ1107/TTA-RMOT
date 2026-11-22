"""One blocking supervisor for strict prefix generation and priority evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT

PRIORITY=('COVER__GRAPH_MEMORY_HMM','COVER__TRACK_SOFT','COVER__FRAME_SOFT',
    'COVER__ENSEMBLE','COVER__RESET_MEMORY_HMM','COVER__GRAPH_ENSEMBLE',
    'COVER__GRAPH_NO_RESET_HMM','UNIFORM__TRACK_SOFT','UNIFORM__GRAPH_MEMORY_HMM')


def generate(pair,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    path=OUT/'supervisor'/('strict_prefix__'+pair+'.log');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,str(ROOT/'scripts/a100_visual_classical_strict_prefix.py'),'--pair',pair],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,(pair,code,str(path))
    record=json.loads((OUT/'strict_prefix'/pair/'completion.json').read_text())
    assert record['GT_open_attempts']==record['forbidden_read_attempts']==0
    return dict(pair=pair,exit_code=code,methods=len(record['methods']))


def evaluate(arguments):
    pair,method=arguments
    from scripts.a100_mechanism_offline import evaluate as official
    folder=OUT/'strict_prefix'/pair/method
    return dict(pair=pair,method=method,folder=str(folder),evaluation=official(folder,pair))


if __name__=='__main__':
    begin=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed=list(pool.map(lambda p:generate(*p),[('C4',2),('C1',3)]))
    atomic_json(OUT/'strict_prefix/generation_completion.json',dict(status='ALL_STRICT_PREFIX_GENERATED',results=completed))
    tasks=[(pair,name) for name in PRIORITY for pair in ('C4','C1')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(evaluate,tasks))
    atomic_json(OUT/'strict_prefix/priority_official_evaluation.json',dict(status='PASS_PRIORITY_FULL_DATA_EVALUATION',
        results=results,wall_seconds=time.perf_counter()-begin,remaining_registered_predictions_preserved=True))
