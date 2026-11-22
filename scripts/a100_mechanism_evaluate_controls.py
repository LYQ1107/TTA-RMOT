"""Post-seal official evaluation entry point for the new controls."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.a100.mechanism import OUT
from scripts.a100_mechanism_offline import evaluate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['temporal','protocol','bank'],required=True);p.add_argument('--name');a=p.parse_args()
    if a.group=='temporal':
        for name in ('RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN'):evaluate(OUT/'temporal/C4'/name,'C4')
    elif a.group=='protocol':evaluate(OUT/'fixed_replay/C4/DS__UNIFIED_CHECKLIST','C4')
    else:evaluate(OUT/'fixed_replay/C4'/a.name,'C4')
