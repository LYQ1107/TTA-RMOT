"""Fixed decoder test, all baselines equally protected, all outcomes retained."""
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


def generate(item):
    pair,seed=item
    with (OUT/'supervisor'/('recovery_contract_'+pair+'_'+str(seed)+'.log')).open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_contract_prefix.py','--pair',pair,'--seed',str(seed)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    assert code==0,(pair,seed,code)


if __name__=='__main__':
    for seed in (0,17,42):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(generate,[(p,seed) for p in ('C4','C1')]))
        group='strong_cover_seed'+str(seed)+'_contract'
        with (OUT/'supervisor'/('recovery_contract_evaluate_'+str(seed)+'.log')).open('a',buffering=1) as log:
            code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group',group,'--primary','COHORT','--workers','4'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        assert code==0,seed
        while not (OUT/('strong_cover_seed'+str(seed))/'official_evaluation.json').is_file():time.sleep(15)
        from scripts.a100_visual_classical_analyze import read_stats
        from scripts.a100_mechanism_analyze import compare_official
        results={}
        for pair in ('C4','C1'):
            record=json.loads((OUT/group/pair/'completion.json').read_text());methods=[r['method'] for r in record['methods']]
            current={k:read_stats(OUT/group/pair/k) for k in methods};previous={k:read_stats(OUT/('strong_cover_seed'+str(seed))/pair/k) for k in methods}
            comparisons={'FULL_vs_'+k:compare_official(current['COHORT'],current[k]) for k in methods if k!='COHORT'}
            effect={k:compare_official(current[k],previous[k]) for k in methods}
            results[pair]=dict(HOTA={k:v['HOTA'] for k,v in current.items()},comparisons=comparisons,decoder_only_effect=effect,
                changed_queries={k:sum(len(v['changes'][k]) for v in record['receipts']) for k in methods},
                decoder_only_CPU_seconds={k:sum(v['decoder_only_seconds'][k] for v in record['receipts']) for k in methods},
                teacher_scores_ranking_and_weights_unchanged=True,GT_used_only_for_postseal_evaluation=True,
                seed=seed,all_development_data=True,descriptive_intervals_not_corrected_for_research_adaptivity=True)
            print(json.dumps(dict(event='TASK_CONTRACT_RESULT',pair=pair,seed=seed,HOTA=results[pair]['HOTA'],
                full_decoder_effect=effect['COHORT'])),flush=True)
        atomic_json(OUT/'analysis'/('contract_seed'+str(seed)+'.json'),results)
