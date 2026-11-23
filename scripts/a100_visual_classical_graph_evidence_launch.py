"""Teacher-guided graph compatibility and no-Yes propagation guard ablations."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_geometry_launch import generate_one,evaluate_one


def main():
    started=time.perf_counter();state=OUT/'supervisor/graph_evidence_state.json'
    specs=[(pair,seed,bounded,True) for bounded in (False,True) for seed in (0,17,42) for pair in ('C4','C1')]
    atomic_json(state,dict(status='GENERATING_ALL_TEACHER_GRAPH_EVIDENCE',pid=os.getpid(),completed=[]))
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(generate_one,specs):
            completed.append(row);atomic_json(state,dict(status='GENERATING_ALL_TEACHER_GRAPH_EVIDENCE',pid=os.getpid(),completed=completed))
    # Every proposed prediction and all quantities are sealed before any GT.
    atomic_json(state,dict(status='EVALUATING_ALL_SEALED_TEACHER_GRAPH_EVIDENCE',pid=os.getpid()))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap();jobs=[(str(Path(r['root'])/m['method']),r['pair']) for r in completed for m in r['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:evaluations=dict(pool.map(evaluate_one,jobs))
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_visual_classical_repeats_analyze import paired
    results={};sufficient={}
    for bounded in (False,True):
        name='bounded' if bounded else 'raw';results[name]={};sufficient[name]={}
        for pair in ('C4','C1'):
            data={};per_seed={};sufficient[name][pair]={};weight_stats=[]
            for seed in (0,17,42):
                group='strong_cover_seed'+str(seed)+('_clipped' if bounded else '')
                root=OUT/'graph_evidence_controls'/group/pair
                receipt=json.loads((root/'completion.json').read_text());per_seed[str(seed)]={}
                folders={r['method']:root/r['method'] for r in receipt['methods']}
                folders.update({key:OUT/group/pair/value for key,value in [
                    ('FULL_VISUAL','COHORT'),('KD','SOFT_KD'),('NO_GRAPH','NO_GRAPH'),
                    ('SOFT_400','SOFT_KD_400'),('HARD_400','HARD_KD_400'),('HARD_FINAL','HARD_KD')]})
                folders['GLOBAL']=OUT/'geometry_controls'/group/pair/'COMPLETE_GRAPH'
                count_family=group+('_counts' if pair=='C4' and not bounded else '_counts_exclusive')
                folders['KD_AT_FULL_COUNT']=OUT/count_family/pair/'SOFT_KD__AT_COHORT_COUNT'
                sufficient[name][pair][str(seed)]={}
                for key,folder in folders.items():
                    item=read_stats(folder);data.setdefault(key,[]).append(item);per_seed[str(seed)][key]=item['HOTA']
                    sufficient[name][pair][str(seed)][key]=dict(parents=item['parent_videos'],numerator=item['parent_stats'][0].tolist(),denominator=item['parent_stats'][1].tolist())
                records=[a for v in receipt['videos'] for a in v['model_evidence']]
                weight_stats.append(dict(seed=seed,eligible_queries=len(records),guard_queries=sum(r['guard_active'] for r in records),
                    mean_model_weights=np.mean([r['weights'] for r in records],axis=0).tolist(),
                    map_choice_counts=np.bincount([np.argmax(r['weights']) for r in records],minlength=3).tolist()))
            contrasts={}
            pairs=[('BMA_GUARD',reference) for reference in ('FULL_VISUAL','KD','NO_GRAPH','GLOBAL','VISUAL_GUARD','BMA','EQUAL_GUARD','SOFT_400','HARD_400','HARD_FINAL')]
            pairs += [('VISUAL_GUARD','FULL_VISUAL'),('BMA','EQUAL'),('MAP_GUARD','BMA_GUARD'),
                      ('BMA_GUARD_AT_FULL_COUNT','FULL_VISUAL'),('BMA_GUARD_AT_FULL_COUNT','KD_AT_FULL_COUNT'),
                      ('BMA_GUARD_AT_FULL_COUNT','EQUAL_GUARD_AT_FULL_COUNT'),('BMA_GUARD_AT_KD_COUNT','KD')]
            for a,b in pairs:contrasts[a+'_vs_'+b]=paired(data[a],data[b])
            probe=json.loads((OUT/'graph_evidence_controls/strong_cover_seed0'/pair/'teacher_only_probe.json').read_text())
            results[name][pair]=dict(per_seed_HOTA=per_seed,mean_HOTA={m:float(np.mean([r[m] for r in per_seed.values()])) for m in per_seed['0']},
                contrasts=contrasts,weights=weight_stats,teacher_only_LOO=probe,primary='BMA_GUARD',
                primary_not_selected_by_target_GT=True,no_target_GT_in_generation=True,no_new_teacher_calls=True,
                head_outputs_identical_to_existing_causal_KD=True,all_development_data=True,
                descriptive_intervals_not_corrected_for_research_adaptivity=True)
            print(json.dumps(dict(event='TEACHER_GRAPH_EVIDENCE_RESULT',labels=name,pair=pair,mean_HOTA=results[name][pair]['mean_HOTA'],
                primary_vs_KD=contrasts['BMA_GUARD_vs_KD'],primary_vs_visual=contrasts['BMA_GUARD_vs_FULL_VISUAL'],
                primary_vs_equal=contrasts['BMA_GUARD_vs_EQUAL_GUARD'],teacher_probe=probe)),flush=True)
    atomic_json(OUT/'analysis/teacher_graph_evidence.json',results)
    atomic_json(OUT/'analysis/teacher_graph_evidence_sufficient_statistics.json',sufficient)
    atomic_json(state,dict(status='PASS_TEACHER_GRAPH_EVIDENCE_STUDY',pid=os.getpid(),wall_seconds=time.perf_counter()-started))


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'supervisor/graph_evidence_state.json',dict(status='FAILED_TEACHER_GRAPH_EVIDENCE_REQUIRES_ROOT_CAUSE',pid=os.getpid()))
        raise
