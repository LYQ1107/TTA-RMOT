"""C1 task-valid fixed-count diagnostics, no adaptation or GT-count selection."""
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


def evaluate_method(folder):
    from flex_uda.a100.evaluation_reuse import evaluate
    return str(folder),evaluate(folder,'C1')


def main():
    for seed in (0,17,42):
        group='strong_cover_seed'+str(seed)
        while not (OUT/group/'C1/completion.json').is_file():time.sleep(15)
        with (OUT/'supervisor'/('recovery_exclusive_counts_'+str(seed)+'.log')).open('a',buffering=1) as log:
            code=subprocess.call([sys.executable,'scripts/a100_visual_classical_strong_countmatched.py','--pair','C1','--group',group,
                '--exclusive-projection'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        assert code==0,seed
        from flex_uda.a100.evaluation_reuse import bootstrap
        bootstrap();family=group+'_counts_exclusive';root=OUT/family/'C1'
        registered=json.loads((root/'completion.json').read_text());assert registered['GT_open_attempts']==0
        original=json.loads((OUT/group/'C1/COHORT/predictions_seal.json').read_text())
        projected=json.loads((root/'COHORT__AT_COHORT_COUNT/predictions_seal.json').read_text())
        before={(r['video'],r['expression']):r['prediction_sha256'] for r in original['episodes']}
        self_changes=sum(r['prediction_sha256']!=before[r['video'],r['expression']] for r in projected['episodes'])
        folders=[root/r['method'] for r in registered['methods']]
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:results=dict(pool.map(evaluate_method,folders))
        atomic_json(OUT/family/'official_evaluation.json',dict(status='PASS_EXCLUSIVE_COUNTMATCHED_C1',results=results,seed=seed))
        from scripts.a100_visual_classical_analyze import read_stats
        from scripts.a100_mechanism_analyze import compare_official
        stats={p.name:read_stats(p) for p in folders};full='COHORT__AT_COHORT_COUNT';comp={}
        for method in ('SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400','NO_GRAPH','STRONG_PRIOR'):
            comp['COUNT_FULL_vs_'+method]=compare_official(stats[full],stats[method+'__AT_COHORT_COUNT'])
        for method in ('SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400'):
            comp['COUNT_'+method]=compare_official(stats['COHORT__AT_'+method+'_COUNT'],stats[method+'__AT_'+method+'_COUNT'])
        atomic_json(OUT/'analysis'/('exclusive_counts_seed'+str(seed)+'.json'),dict(HOTA={k:v['HOTA'] for k,v in stats.items()},comparisons=comp,
            seed=seed,pair='C1',target_GT_counts_never_used=True,exclusivity_preserved=True,all_development_data=True,
            full_countself_changed_query_files=self_changes,full_countself_total_query_files=len(projected['episodes']),
            confidence_descriptive_not_corrected_for_research_adaptivity=True))
        print(json.dumps(dict(event='EXCLUSIVE_COUNTMATCHED_COMPLETE',seed=seed,comparisons=comp)),flush=True)


if __name__=='__main__':main()
