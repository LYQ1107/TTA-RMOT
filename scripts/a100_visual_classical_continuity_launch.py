"""Blocking generation/evaluation supervisor for immutable ID-only controls."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def generate(pair):
    path=OUT/'supervisor'/('recovery_continuity_'+pair+'.log')
    with path.open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_continuity.py','--pair',pair],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    assert code==0,(pair,code,str(path))
    r=json.loads((OUT/'continuity_prefix'/pair/'completion.json').read_text())
    assert r['GT_open_attempts']==r['forbidden_read_attempts']==0 and r['same_boxes_and_counts_verified']


if __name__=='__main__':
    begin=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(generate,['C4','C1']))
    logpath=OUT/'supervisor/recovery_continuity_evaluate.log'
    with logpath.open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group','continuity_prefix',
            '--primary','EXEMPLAR__VISUAL_SEMANTIC','--workers','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(OUT/'continuity_prefix/supervisor_completion.json',dict(exit_code=code,wall_seconds=time.perf_counter()-begin))
    assert code==0
