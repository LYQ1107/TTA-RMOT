"""Public-only all93 HOTA/all90 effect reproduction for this exact cohort."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_primary_statistics_common import collect,reproduce

STEM='crossfit_pattern_teacher_primary_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'/STEM

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');parser.add_argument('--reproduce',action='store_true')
    args=parser.parse_args()
    if args.collect: collect(OUT,PUBLIC,STEM,93)
    else: reproduce(PUBLIC,STEM,93,90)
