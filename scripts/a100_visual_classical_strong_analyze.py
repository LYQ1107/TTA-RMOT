"""Descriptive post-seal comparisons against every strengthened KD baseline."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_mechanism_analyze import compare_official


def analyze(seed=0,policies=('cover','uniform')):
    groups=['strong_'+p+'_seed'+str(seed) for p in policies]
    while not all((OUT/g/'official_evaluation.json').is_file() and (OUT/(g+'_counts')/'official_evaluation.json').is_file() for g in groups):time.sleep(15)
    result={}
    for pair in ('C4','C1'):
        data={};comparisons={};cost={}
        for group in groups:
            for family in (group,group+'_counts'):
                for row in json.loads((OUT/family/'official_evaluation.json').read_text())['results']:
                    if row['pair']==pair:data[family+'/'+row['method']]=read_stats(row['folder'])
            def add(name,a,b):comparisons[group+'/'+name]=compare_official(data[a],data[b])
            full=group+'/COHORT';projected=group+'_counts/COHORT__AT_COHORT_COUNT'
            for other in ('SOFT_KD_400','SOFT_KD_4000','SOFT_KD','HARD_KD_400','HARD_KD_4000','HARD_KD',
                          'NO_GRAPH','WHITENED_COHORT','STRONG_PRIOR','PEER','UNIFORM_PRECISION'):
                add('FULL_vs_'+other,full,group+'/'+other)
                add('COUNT_FULL_vs_'+other,projected,group+'_counts/'+other+'__AT_COHORT_COUNT')
            for label in ('SOFT_KD','HARD_KD'):
                add('COUNT_'+label,group+'_counts/COHORT__AT_'+label+'_COUNT',group+'_counts/'+label+'__AT_'+label+'_COUNT')
            record=json.loads((OUT/group/pair/'completion.json').read_text());assert record['GT_open_attempts']==record['forbidden_read_attempts']==0
            cost[group]={label:dict(wall_seconds=sum(v['fits'][label]['wall_seconds'] for v in record['receipts']),
                maximum_teacher_KL=max(v['fits'][label]['teacher_KL'] for v in record['receipts']),
                videos_exceeding_fit_tolerance=sum(not v['fits'][label]['teacher_fit_within_0_005'] for v in record['receipts'])) for label in ('SOFT','HARD')}
        if len(groups)>1:comparisons['COVER_vs_UNIFORM']=compare_official(data[groups[0]+'/COHORT'],data[groups[1]+'/COHORT'])
        result[pair]=dict(HOTA={k:v['HOTA'] for k,v in data.items()},comparisons=comparisons,fit_cost=cost,
            seed=seed,no_target_GT_in_adaptation=True,all_development_data=True,
            intervals_descriptive_not_adjusted_for_research_adaptivity=True,no_independent_blind_test=True,
            previously_cached_images_features_and_QA_not_a_cold_live_cost_benchmark=True)
        print(json.dumps(dict(event='STRONG_BASELINE_ANALYSIS',pair=pair,HOTA={k:v['HOTA'] for k,v in data.items() if '_counts/' not in k},
            primary_comparisons={k:v for k,v in comparisons.items() if k.endswith(('FULL_vs_SOFT_KD','FULL_vs_HARD_KD','FULL_vs_NO_GRAPH','COUNT_FULL_vs_SOFT_KD'))})),flush=True)
    atomic_json(OUT/'analysis'/('strong_prefix_seed'+str(seed)+'.json'),result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--policies',nargs='+',default=['cover','uniform'])
    a=p.parse_args();analyze(a.seed,a.policies)
