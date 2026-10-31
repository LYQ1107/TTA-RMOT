"""Post-seal target evaluation; old X0 and three references are exactly reused."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *
from scripts import evaluate_diag_attrib_v2 as evaluator

def before_GT():
    setup();require_plan();s=json.loads((RUN/'predictions_seal.json').read_text());p=json.loads((RUN/'teacher_combo_seal.json').read_text())
    if s['status']!='ALL36_GT_FREE_PREDICTIONS_SEALED' or s['target_GT_open_attempts'] or p['target_GT_open_attempts']:raise ValueError('seal all GT-free predictions before GT')
    for r in s['methods']:v2.verify_seal(r['method'])
    if file_sha(p['outputs_path'])!=p['outputs_sha256']:raise ValueError('teacher combo changed after sealing')
    return s

def evaluate():
    started=time.perf_counter();s=before_GT();meta,_,_=inputs(verify=False)
    refs={'FROZEN':v2.RUN/'evaluation/M_FROZEN__FROZEN','B1_SRC_RATE_GLOBAL':v2.RUN/'evaluation/B1_SRC_RATE_GLOBAL',
          'OFFICIAL_DANCE_COMPLETE':v2.RUN/'evaluation/OFFICIAL_DANCE_COMPLETE'}
    for m,original in refs.items():
        selected={(q['video_id'],q['expression_id']):(original/'results'/q['video_id']/q['expression_id']/'predict.txt').read_text().splitlines() for q in meta['episodes']}
        v2.seal_plain(m,meta['episodes'],selected,protocol='EXACT_REUSED_SEALED_V2_BASELINE');v2.reuse_metrics(m,original)
    reused=list(refs)
    for m in cs.MARGINS:
        name=m+'__X0_C3';v2.reuse_metrics(name,cs.RUN/'evaluation'/(m+'__C3_QUERY_ARGMAX_TRACK'));reused.append(name)
    evaluator.RUN=RUN;evaluator.EVAL=v2.EVAL;evaluator.require_prereg=require_plan
    methods=[r['method'] for r in s['methods']]+list(refs);new=[m for m in methods if m not in reused]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluator.evaluate,m) for m in new]):f.result()
    record=dict(status='ALL39_OFFICIAL_TARGET_SETS_COMPLETE',new_jobs=len(new),reused_jobs=len(reused),
        methods={m:json.loads((v2.EVAL/m/'official_completion.json').read_text()) for m in methods},actual_batch_wall_seconds=time.perf_counter()-started)
    atomic_json(RUN/'official_completion.json',record);atomic_json(PUBLIC/'official_completion.json',record)
    print(json.dumps(dict(event='XQUERY_ALL_TARGET_OFFICIAL_COMPLETE',new=len(new),reused=len(reused))),flush=True)

if __name__=='__main__':evaluate()
