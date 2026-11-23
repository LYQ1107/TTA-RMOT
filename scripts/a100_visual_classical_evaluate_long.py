"""Isolated evaluation of the stronger teacher-fit baseline, after V1 seals."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def run(pair):
    for p in ('C4','C1'):
        assert (OUT/'predictions'/p/'frame_kd_completion.json').is_file()
        assert (OUT/'countmatched'/p/'completion.json').is_file()
        assert (OUT/'convergence'/p/'completion.json').is_file()
    from scripts.a100_mechanism_offline import evaluate
    begin=time.perf_counter();source=OUT/'convergence'/pair
    rows=json.loads((source/'completion.json').read_text())['methods'];results=[]
    for row in rows:
        folder=source/row['method'];record=evaluate(folder,pair)
        results.append(dict(method=row['method'],folder=str(folder),evaluation=record))
    atomic_json(source/'official_evaluation.json',dict(status='PASS_OFFICIAL_CONVERGENCE_BASELINES',
        pair=pair,results=results,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();run(a.pair)
