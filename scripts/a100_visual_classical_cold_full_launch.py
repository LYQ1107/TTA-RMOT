"""Wait for this goal's GPU1 jobs, then measure the whole fresh C4 stream."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def evaluate_one(folder):
    from flex_uda.a100.evaluation_reuse import evaluate
    return str(folder),evaluate(folder,'C4')


def main(name='C4_fresh_full_stream_seed0',gpu=1,perception_mode='full',skip_own_queue=False):
    started=time.perf_counter();root=OUT/'cold_full_c4'/name
    tag='cold_full' if perception_mode=='full' else 'cold_sparse'
    state=OUT/'supervisor'/(tag+'_state.json')
    while not skip_own_queue:
        fields=OUT/'supervisor/field_state.json';clip=OUT/'supervisor/clipped_repeats_state.json'
        a=json.loads(fields.read_text()) if fields.is_file() else {};b=json.loads(clip.read_text()) if clip.is_file() else {}
        free_field=a.get('jobs',{}).get('field_fit_C1_FIELD_STATIC',{}).get('exit_code')==0
        free_clip=b.get('jobs',{}).get('clip_repeat_C1_42',{}).get('exit_code')==0
        atomic_json(state,dict(status='WAITING_FOR_OWN_GPU1_JOBS',pid=os.getpid(),field_GPU1_finished=free_field,
            clipped_GPU1_finished=free_clip,queue_wait_seconds=time.perf_counter()-started))
        if free_field and free_clip:break
        time.sleep(15)
    queued=time.perf_counter()-started
    atomic_json(state,dict(status='RUNNING_FRESH_WHOLE_C4',pid=os.getpid(),physical_GPU=gpu,source_perception_mode=perception_mode,queue_wait_seconds=queued))
    log_path=OUT/'supervisor'/('recovery_strong_field_cold_full'+('_sparse' if perception_mode!='full' else '')+'_worker.log')
    with log_path.open('a',buffering=1) as log:
        begin=time.perf_counter();child=subprocess.Popen([sys.executable,'scripts/a100_visual_classical_cold_full_c4.py','--name',name,'--perception-mode',perception_mode],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        atomic_json(OUT/'supervisor'/(tag+'_worker.json'),dict(pid=child.pid,log=str(log_path),physical_GPU=gpu))
        code=child.wait();wall=time.perf_counter()-begin
    atomic_json(root/'worker_process_receipt.json',dict(exit_code=code,wall_seconds=wall,queue_wait_excluded_seconds=queued))
    if code:
        p=root/'phase_ledger.json'
        if p.is_file():
            ledger=json.loads(p.read_text());ledger['attempts'][-1].update(observed_worker_wall_seconds=wall,exit_code=code)
            atomic_json(p,ledger)
        atomic_json(state,dict(status='FAILED_FRESH_C4_REQUIRES_ROOT_CAUSE',pid=os.getpid(),exit_code=code,worker_wall_seconds=wall))
        raise RuntimeError('fresh C4 worker failed: '+str(code))
    receipt=json.loads((root/'completion.json').read_text());assert receipt['GT_open_attempts']==0
    atomic_json(state,dict(status='EVALUATING_FRESH_WHOLE_C4',pid=os.getpid(),physical_GPU=gpu,worker_wall_seconds=wall))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap();folders=[root/'methods'/r['method'] for r in receipt['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:evaluated=dict(pool.map(evaluate_one,folders))
    results={Path(k).name:v for k,v in evaluated.items()}
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_mechanism_analyze import compare_official
    data={method:read_stats(root/'methods'/method) for method in results};comparisons={}
    for label in ('RAW','CLIP'):
        full=label+'_COHORT'
        for baseline in [k for k in results if '_KD' in k or k.endswith('_MEAN') or k==label+'_NO_GRAPH' or k==label+'_STRONG_PRIOR']:
            comparisons[full+'_vs_'+baseline]=compare_official(data[full],data[baseline])
        for baseline in (label+'_MEAN',label+'_MEAN_NO_GRAPH'):
            comparisons[label+'_MEAN_COHORT_vs_'+baseline]=compare_official(data[label+'_MEAN_COHORT'],data[baseline])
    report=dict(status='PASS_FRESH_WHOLE_C4_PIPELINE_AND_OFFICIAL',HOTA={k:v['HOTA_unrounded'] for k,v in results.items()},
        comparisons=comparisons,cost_by_method=receipt['cost_by_method'],logical_teacher_calls=receipt['logical_teacher_calls'],
        actual_teacher=receipt['teacher'],single_uninterrupted_attempt=receipt['single_uninterrupted_attempt'],
        whole_dataset=receipt['protocol'],all_development_data=True,shared_physical_work_not_independent_cold_arms=True,
        total_experiment_wall_seconds=time.perf_counter()-started,queue_wait_seconds=queued)
    analysis_name='fresh_whole_C4.json' if perception_mode=='full' else 'fresh_whole_C4_sparse.json'
    atomic_json(root/'official_evaluation.json',report);atomic_json(OUT/'analysis'/analysis_name,report)
    atomic_json(state,dict(status='PASS_FRESH_WHOLE_C4',pid=os.getpid(),worker_wall_seconds=wall))
    print(json.dumps(dict(event='FRESH_WHOLE_C4_COMPLETE',HOTA=report['HOTA'],actual_teacher_calls=receipt['teacher']['new_calls'],
        cache_hits=receipt['teacher']['cache_hits'])),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='C4_fresh_full_stream_seed0');p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--perception-mode',choices=['full','track-three'],default='full');p.add_argument('--skip-own-queue',action='store_true')
    a=p.parse_args();main(a.name,a.gpu,a.perception_mode,a.skip_own_queue)
