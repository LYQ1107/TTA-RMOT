"""Persistent fixed-answer teacher-audit supervisor; no GPU models are loaded."""
import concurrent.futures
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT


def generate(pair):
    with (OUT/'supervisor'/('recovery_peer_prefix_'+pair+'.log')).open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_peer_prefix.py','--pair',pair],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,(pair,code)


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(generate,['C4','C1']))
    with (OUT/'supervisor/recovery_peer_evaluate.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group','peer_prefix',
            '--primary','COVER__AUDIT','--workers','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert code==0
