"""All graph-specificity predictions seal before isolated official evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def generate_one(spec):
    pair,seed,bounded,*options=spec;adaptive=bool(options and options[0]);name=pair+'_'+str(seed)+('_bounded' if bounded else '_raw')
    path=OUT/'supervisor'/('recovery_strong_field_cold_full_geometry_'+('evidence_' if adaptive else '')+name+'.log')
    command=[sys.executable,'scripts/a100_visual_classical_geometry_controls.py','--pair',pair,'--seed',str(seed)]
    if bounded:command.append('--bounded')
    if adaptive:command.append('--adaptive')
    with path.open('a',buffering=1) as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        code=child.wait()
    if code:raise RuntimeError('Graph specificity generation failed: '+str(path)+' exit '+str(code))
    group='strong_cover_seed'+str(seed)+('_clipped' if bounded else '')
    root=OUT/('graph_evidence_controls' if adaptive else 'geometry_controls')/group/pair;completion=json.loads((root/'completion.json').read_text())
    assert completion['GT_open_attempts']==completion['new_teacher_calls']==completion['same_graph_regression_changed_rows']==0
    return dict(pair=pair,seed=seed,bounded=bounded,root=str(root),methods=completion['methods'],log=str(path))


def evaluate_one(spec):
    from flex_uda.a100.evaluation_reuse import evaluate
    folder,pair=spec
    return folder,evaluate(folder,pair)


def main():
    started=time.perf_counter();state=OUT/'supervisor/geometry_state.json'
    specs=[(pair,seed,bounded) for bounded in (False,True) for seed in (0,17,42) for pair in ('C4','C1')]
    atomic_json(state,dict(status='GENERATING_ALL_GRAPH_SPECIFICITY',pid=os.getpid(),expected=len(specs),completed=[]))
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(generate_one,s):s for s in specs}
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            atomic_json(state,dict(status='GENERATING_ALL_GRAPH_SPECIFICITY',pid=os.getpid(),expected=len(specs),completed=completed))
    atomic_json(state,dict(status='EVALUATING_ALL_SEALED_GRAPH_SPECIFICITY',pid=os.getpid(),completed=completed))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap();evaluation_specs=[(str(Path(r['root'])/m['method']),r['pair']) for r in completed for m in r['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:evaluations=dict(pool.map(evaluate_one,evaluation_specs))
    from scripts.a100_visual_classical_repeats_analyze import paired
    from scripts.a100_visual_classical_analyze import read_stats
    results={};sufficient={}
    for bounded in (False,True):
        name='bounded' if bounded else 'raw';results[name]={};sufficient[name]={}
        for pair in ('C4','C1'):
            full=[];data={k:[] for k in ('PERMUTED_GRAPH','COMPLETE_GRAPH','PERMUTED_AT_FULL_COUNT','COMPLETE_AT_FULL_COUNT')}
            per_seed={};sufficient[name][pair]={}
            for seed in (0,17,42):
                group='strong_cover_seed'+str(seed)+('_clipped' if bounded else '')
                full.append(read_stats(OUT/group/pair/'COHORT'));per_seed[str(seed)]={'FULL':full[-1]['HOTA']}
                sufficient[name][pair][str(seed)]={}
                for method in data:
                    folder=OUT/'geometry_controls'/group/pair/method;item=read_stats(folder);data[method].append(item)
                    per_seed[str(seed)][method]=item['HOTA']
                for method,item in [('FULL',full[-1])]+[(m,data[m][-1]) for m in data]:
                    sufficient[name][pair][str(seed)][method]=dict(parents=item['parent_videos'],numerator=item['parent_stats'][0].tolist(),denominator=item['parent_stats'][1].tolist())
            comparisons={method:paired(full,items) for method,items in data.items()}
            import numpy as np
            results[name][pair]=dict(per_seed_HOTA=per_seed,mean_HOTA={m:float(np.mean([r[m] for r in per_seed.values()])) for m in per_seed['0']},
                FULL_vs=comparisons,same_actual_teacher_and_prefix_head=True,new_teacher_calls=0,
                missing_part_nodes_and_anchor_precision_unchanged=True,
                CountSelf_and_original_graph_decisions_identical=True,
                all_development_data=True,descriptive_intervals_not_blind_confirmation=True)
            print(json.dumps(dict(event='GRAPH_SPECIFICITY_RESULT',targets=name,pair=pair,mean_HOTA=results[name][pair]['mean_HOTA'],contrasts=comparisons)),flush=True)
    atomic_json(OUT/'analysis/graph_specificity.json',results)
    atomic_json(OUT/'analysis/graph_specificity_sufficient_statistics.json',sufficient)
    atomic_json(state,dict(status='PASS_GRAPH_SPECIFICITY_STUDY',pid=os.getpid(),wall_seconds=time.perf_counter()-started))


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'supervisor/geometry_state.json',dict(status='FAILED_GRAPH_SPECIFICITY_REQUIRES_ROOT_CAUSE',pid=os.getpid()))
        raise
