"""Same .02 teacher softening in an actual trained KD baseline and classic arm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


if __name__=='__main__':
    jobs=[];group='strong_cover_seed0_clipped';begin=time.perf_counter()
    for pair in ('C4','C1'):
        log=(OUT/'supervisor'/('recovery_clipped_'+pair+'.log')).open('a',buffering=1)
        child=subprocess.Popen([sys.executable,'scripts/a100_visual_classical_strong_prefix.py','--pair',pair,'--policy','COVER',
            '--teacher-target-clip','.02'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        jobs.append((pair,child,log))
    for pair,p,log in jobs:
        code=p.wait();log.close();assert code==0,(pair,code)
    with (OUT/'supervisor/recovery_clipped_evaluate.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_evaluate_group.py','--group',group,'--primary','COHORT',
            '--workers','4'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert code==0
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_mechanism_analyze import compare_official
    results={}
    for pair in ('C4','C1'):
        rows=json.loads((OUT/group/pair/'completion.json').read_text());assert rows['GT_open_attempts']==rows['forbidden_read_attempts']==0
        data={k['method']:read_stats(OUT/group/pair/k['method']) for k in rows['methods']}
        comparisons={'COHORT_vs_'+k:compare_official(data['COHORT'],data[k]) for k in data if k!='COHORT'}
        for method in ('COHORT','SOFT_KD','SOFT_KD_400','HARD_KD'):
            comparisons['clipped_vs_raw_'+method]=compare_official(data[method],read_stats(OUT/'strong_cover_seed0'/pair/method))
        results[pair]=dict(HOTA={k:v['HOTA'] for k,v in data.items()},comparisons=comparisons,teacher_softening=.02,
            actual_KD_training=True,no_hyperparameter_grid=True,seed=0,all_development_data=True,
            no_target_GT_in_adaptation=True,descriptive_intervals_not_research_adaptivity_corrected=True)
        print(json.dumps(dict(event='MATCHED_CLIPPED_TEACHER_RESULT',pair=pair,HOTA=results[pair]['HOTA'],
            full_vs_clipped_soft=comparisons['COHORT_vs_SOFT_KD'])),flush=True)
    atomic_json(OUT/'analysis/matched_clipped_teacher.json',dict(results=results,experiment_wall_seconds=time.perf_counter()-begin,
        actual_soft_KD_receives_same_target_clipping_used_by_graph_anchors=True))
