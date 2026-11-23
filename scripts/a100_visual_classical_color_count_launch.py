"""Same-feature colour KD vs cohort ranking at identical predicted counts."""
import concurrent.futures
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT


def generate(pair):
    with (OUT/'supervisor'/('recovery_colour_counts_'+pair+'.log')).open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_prefix_countmatched.py','--pair',pair,'--family','colour'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,(pair,code)


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(generate,['C4','C1']))
    with (OUT/'supervisor/recovery_colour_counts_evaluate.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group','colour_countmatched',
            '--primary','FULL__AT_FULL_COUNT','--workers','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0
