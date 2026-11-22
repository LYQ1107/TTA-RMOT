"""Independent, parallel official evaluation of disjoint sealed V2 folders."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def job(args):
    pair,method=args
    from scripts.a100_mechanism_offline import evaluate
    folder=OUT/'refinement_v2'/pair/method
    result=evaluate(folder,pair)
    return dict(pair=pair,method=method,folder=str(folder),evaluation=result)


def main():
    begin=time.perf_counter();tasks=[]
    for pair in ('C4','C1'):
        done=json.loads((OUT/'refinement_v2'/pair/'completion.json').read_text())
        assert done['GT_open_attempts']==0
    for policy in ('COVER','UNIFORM','PRIOR','UNCERTAINTY'):
        for mode in ('HIDDEN_GRAPH','RAW_GRAPH','NO_GRAPH'):
            for pair in ('C4','C1'):tasks.append((pair,policy+'__'+mode))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(job,tasks):
            results.append(result)
    atomic_json(OUT/'refinement_v2/official_evaluation.json',dict(status='PASS_V2_ALL_PREDECLARED_UNITS',
        results=results,wall_seconds=time.perf_counter()-begin,parallel_disjoint_folders=True))


if __name__=='__main__':main()
