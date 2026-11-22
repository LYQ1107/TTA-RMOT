"""Isolated post-seal official evaluation. Never imported by adaptation."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_visual_classical_prepare import OUT


def main(pair):
    from scripts.a100_mechanism_offline import evaluate
    begin=time.perf_counter();results=[];identical={}
    # Require both directions and every predeclared arm to be sealed first.
    for required in ('C4','C1'):
        for group in ('predictions','countmatched'):
            assert (OUT/group/required/'completion.json').is_file(),(group,required)
        assert (OUT/'predictions'/required/'frame_kd_completion.json').is_file()
    for group in ('predictions','countmatched'):
        root=OUT/group/pair;completion=json.loads((root/'completion.json').read_text())
        for row in completion['methods']:
            folder=root/row['method'];seal=json.loads((folder/'predictions_seal.json').read_text())
            assert seal['GT_open_attempts']==0
            signature=tuple(q['prediction_sha256'] for q in seal['episodes'])
            if signature in identical:
                previous=identical[signature]
                record=dict(previous['evaluation'],identical_prediction_reference=previous['folder'])
                atomic_json(folder/'official_equivalent.json',record)
            else:
                record=evaluate(folder,pair)
                identical[signature]=dict(folder=str(folder),evaluation=record)
            results.append(dict(group=group,method=row['method'],folder=str(folder),evaluation=record))
    atomic_json(OUT/'evaluation'/pair/'completion.json',dict(status='OFFICIAL_EVALUATION_COMPLETE',pair=pair,
        results=results,wall_seconds=time.perf_counter()-begin,
        target_GT_only_in_this_evaluation_process=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();main(a.pair)
