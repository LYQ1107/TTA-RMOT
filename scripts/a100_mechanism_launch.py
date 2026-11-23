"""Single blocking supervisor; do not interrupt unrelated GPU workloads."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_supervisor import run
from flex_uda.a100.mechanism import OUT


if __name__=='__main__':
    run([str(ROOT/'scripts/a100_mechanism_collect.py')],0,OUT/'evidence')
