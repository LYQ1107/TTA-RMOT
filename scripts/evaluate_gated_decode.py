"""Separate post-seal official evaluators. No decoder thresholds are fitted."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *
from scripts import evaluate_diag_attrib_v2 as evaluator
from scripts.pf_diag_common import official_metrics

def before_GT():
    setup();require_plan();p=json.loads((RUN/'predictions_seal.json').read_text());g=json.loads((RUN/'gates_seal.json').read_text())
    if p['status']!='ALL60_DECODER_SETS_SEALED_BEFORE_GT' or p['GT_open_attempts'] or g['GT_open_attempts']:raise ValueError('finish GT-free work first')
    for field,sha in [('gate_path','gates_sha256'),('exclusivity_path','exclusivity_sha256'),('teacher_output_path','teacher_output_sha256')]:
        if file_sha(g[field])!=g[sha]:raise ValueError('sealed gates/teacher changed')
    if file_sha(RUN/'gates_seal.json')!=p['gates_seal_sha256'] or file_sha(RUN/'source_boundary/predictions_seal.json')!=p['source_seal_sha256']:raise ValueError('changed outer seals')
    for r in p['methods']:v2.verify_seal(r['method'])
    source=json.loads((RUN/'source_boundary/predictions_seal.json').read_text())
    for r in source['methods']:
        if len(r['episodes'])!=862:raise ValueError('source coverage')
        for e in r['episodes']:
            if file_sha(e['prediction_path'])!=e['prediction_sha256']:raise ValueError('source changed')
    return p,source

def signature(manifest):return tuple(e['prediction_sha256'] for e in manifest['episodes'])

def source_job(record):
    folder=RUN/'source_boundary/evaluation'/record['method'];done=folder/'official_completion.json'
    if done.exists():
        r=json.loads(done.read_text())
        if r['status']=='PASS' and file_sha(folder/'results/pedestrian_detailed.csv')==r['detailed_sha256'] and file_sha(folder/'results/pedestrian_summary.txt')==r['summary_sha256']:return r
    old=OLD/'source_boundary/evaluation'/(record['margin']+'__NATIVE')
    same=all(file_sha(old/'results'/e['video']/e['expression']/'predict.txt')==e['prediction_sha256'] for e in record['episodes'])
    start=time.perf_counter()
    if same:
        for filename in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(old/'results'/filename,folder/'results'/filename)
        result=dict(status='PASS',exit_code=0,reused_exact_reference=str(old),new_TrackEval_job=False)
    else:
        for e in record['episodes']:
            link=folder/'results'/e['video']/e['expression']/'gt.txt'
            if not link.exists():link.symlink_to(SOURCE/'gt_template'/e['video']/e['expression']/'gt.txt')
        command=[sys.executable,'./TrackEval/scripts/run_mot_challenge.py','--METRICS','HOTA','CLEAR','Identity',
            '--SEQMAP_FILE','./seqmaps/kitti-2.txt','--SKIP_SPLIT_FOL','True','--GT_FOLDER',str(SOURCE/'KITTI/training/image_02'),
            '--TRACKERS_FOLDER',str(folder/'results'),'--GT_LOC_FORMAT','{gt_folder}{video_id}/{expression_id}/gt.txt',
            '--TRACKERS_TO_EVAL',str(folder/'results'),'--USE_PARALLEL','True','--NUM_PARALLEL_CORES','2','--PLOT_CURVES','False']
        with (folder/'trackeval.log').open('w') as h:code=subprocess.call(command,cwd=ROOT/'vendor/FlexHook',stdout=h,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('source evaluation failed; '+str(folder/'trackeval.log'))
        result=dict(status='PASS',exit_code=0,new_TrackEval_job=True)
    result.update(method=record['method'],margin=record['margin'],decoder=record['decoder'],wall_seconds=time.perf_counter()-start,
        prediction_rows=record['prediction_rows'],metrics=official_metrics(folder/'results'),
        summary_sha256=file_sha(folder/'results/pedestrian_summary.txt'),detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'))
    atomic_json(done,result);print(json.dumps(dict(event='GATED_SOURCE_OFFICIAL_PASS',**result)),flush=True);return result

def evaluate():
    start=time.perf_counter();p,source=before_GT();meta,_,_=inputs()
    refs={'FROZEN':v2.RUN/'evaluation/M_FROZEN__FROZEN','B1_SRC_RATE_GLOBAL':v2.RUN/'evaluation/B1_SRC_RATE_GLOBAL',
        'OFFICIAL_DANCE_COMPLETE':v2.RUN/'evaluation/OFFICIAL_DANCE_COMPLETE'}
    reused=[];canonical={}
    for m,original in refs.items():
        selected={(q['video_id'],q['expression_id']):(original/'results'/q['video_id']/q['expression_id']/'predict.txt').read_text().splitlines() for q in meta['episodes']}
        s=v2.seal_plain(m,meta['episodes'],selected,protocol='EXACT_REUSED_NATIVE_REFERENCE');v2.reuse_metrics(m,original)
        reused.append(m);canonical[signature(s)]=v2.EVAL/m
    for r in p['methods']:
        if r['decoder'] in ('X0_C3','X1_C3_CAUSAL'):
            m=r['method'];s=v2.verify_seal(m);v2.reuse_metrics(m,OLD/'evaluation'/m);reused.append(m);canonical[signature(s)]=v2.EVAL/m
    pending=[];aliases=[]
    for r in p['methods']:
        m=r['method']
        if m in reused:continue
        sig=signature(v2.verify_seal(m))
        if sig in canonical:aliases.append((m,canonical[sig]))
        else:canonical[sig]=v2.EVAL/m;pending.append(m)
    evaluator.RUN=RUN;evaluator.EVAL=v2.EVAL;evaluator.require_prereg=require_plan
    with ThreadPoolExecutor(max_workers=6) as pool:
        # At most four target jobs, at most two source jobs concurrently.
        def target_batch():
            with ThreadPoolExecutor(max_workers=4) as targets:
                for f in as_completed([targets.submit(evaluator.evaluate,m) for m in pending]):f.result()
        jobs=[pool.submit(target_batch),*[pool.submit(source_job,r) for r in source['methods']]]
        for f in as_completed(jobs):f.result()
    for m,original in aliases:v2.reuse_metrics(m,original);reused.append(m)
    names=[r['method'] for r in p['methods']]+list(refs)
    result=dict(status='ALL63_TARGET_AND4_SOURCE_OFFICIAL_COMPLETE',new_target_jobs=len(pending),reused_target_jobs=len(reused),
        methods={m:json.loads((v2.EVAL/m/'official_completion.json').read_text()) for m in names},
        source_methods={r['method']:json.loads((RUN/'source_boundary/evaluation'/r['method']/'official_completion.json').read_text()) for r in source['methods']},
        actual_batch_wall_seconds=time.perf_counter()-start)
    atomic_json(RUN/'official_completion.json',result);atomic_json(PUBLIC/'official_completion.json',result)
    print(json.dumps(dict(event='GATED_ALL_OFFICIAL_COMPLETE',new_target_jobs=len(pending),reused_target_jobs=len(reused),wall_seconds=result['actual_batch_wall_seconds'])),flush=True)

if __name__=='__main__':evaluate()
