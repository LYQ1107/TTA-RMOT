"""Post-seal official evaluation, original v2 numeric/metric code unchanged."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import *
from scripts import evaluate_diag_attrib_v2 as evaluator
from scripts.count_structure_oracle import before_GT

def evaluate():
    setup();before_GT();oracle=json.loads((RUN/'oracle_seal.json').read_text())
    if oracle['status']!='ALL6_ORACLE_SEALED_F_FRAME_REPRODUCED':raise ValueError('stop: oracle self-check not passed')
    refs={'FROZEN':OLD/'evaluation/M_FROZEN__FROZEN','B1_SRC_RATE_GLOBAL':OLD/'evaluation/B1_SRC_RATE_GLOBAL'}
    meta,_,groups=load_inputs(verify=False)
    for m,original in refs.items():
        selected={(q['video_id'],q['expression_id']):(original/'results'/q['video_id']/q['expression_id']/'predict.txt').read_text().splitlines() for q in meta['episodes']}
        v2.seal_plain(m,meta['episodes'],selected,protocol='EXACT_REUSED_V2_REFERENCE');v2.reuse_metrics(m,original)
    v2.reuse_metrics('ORACLE__F_FRAME',OLD/'evaluation/M_FROZEN__ORACLE_GT')
    selectors=json.loads((RUN/'selectors_seal.json').read_text())
    methods=[*oracle['methods'],*[r['method'] for r in selectors['methods']],*refs]
    evaluator.RUN=RUN;evaluator.EVAL=v2.EVAL;evaluator.require_prereg=require_plan
    reused={*refs,'ORACLE__F_FRAME'};new=[m for m in methods if m not in reused]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluator.evaluate,m) for m in new]):f.result()
    results={m:json.loads((v2.EVAL/m/'official_completion.json').read_text()) for m in methods}
    record=dict(status='ALL36_OFFICIAL_PREDICTION_SETS_COMPLETE',methods=results,new_jobs=len(new),reused_jobs=len(reused),
        source_or_student_training=False,original_v2_metric_code_unchanged=True)
    atomic_json(RUN/'official_completion.json',record);atomic_json(PUBLIC/'official_completion.json',record)
    print(json.dumps(dict(event='COUNT_OFFICIAL_ALL_COMPLETE',new_jobs=len(new),reused_jobs=len(reused))),flush=True)

if __name__=='__main__':evaluate()
