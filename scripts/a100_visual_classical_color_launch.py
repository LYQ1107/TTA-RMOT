"""One persistent supervisor for image-cohort prefix adaptation and metrics."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT


def generate(arguments):
    pair,gpu=arguments
    with (OUT/'supervisor'/('recovery_colour_prefix_'+pair+'.log')).open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_color_prefix.py','--pair',pair],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    assert code==0,(pair,code)


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(generate,[('C4',0),('C1',1)]))
    with (OUT/'supervisor/recovery_colour_evaluate.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group','colour_prefix',
            '--primary','COVER__COHORT','--workers','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0
