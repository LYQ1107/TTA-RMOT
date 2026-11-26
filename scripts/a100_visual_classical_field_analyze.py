"""Fixed-budget adaptive feedback attribution after both prediction sets seal."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_mechanism_analyze import compare_official


if __name__=='__main__':
    results={}
    for pair in ('C4','C1'):
        data={};comparisons={};budgets={}
        for policy in ('FIELD','FIELD_STATIC','COVER','UNIFORM'):
            group='strong_'+policy.lower()+'_seed0';record=json.loads((OUT/group/pair/'completion.json').read_text())
            assert record['GT_open_attempts']==record['forbidden_read_attempts']==0
            budgets[policy]=record['receipts'][-1]['teacher_examples']
            for item in record['methods']:data[policy+'/'+item['method']]=read_stats(OUT/group/pair/item['method'])
            if policy.startswith('FIELD'):
                family=group+'_counts_exclusive'
                for item in json.loads((OUT/family/pair/'completion.json').read_text())['methods']:
                    data[policy+'/counts/'+item['method']]=read_stats(OUT/family/pair/item['method'])
        assert len(set(budgets.values()))==1
        for policy in ('FIELD','FIELD_STATIC'):
            for method in ('SOFT_KD_400','SOFT_KD','HARD_KD_400','HARD_KD','NO_GRAPH','STRONG_PRIOR'):
                comparisons[policy+'/FULL_vs_'+method]=compare_official(data[policy+'/COHORT'],data[policy+'/'+method])
                comparisons[policy+'/COUNT_FULL_vs_'+method]=compare_official(data[policy+'/counts/COHORT__AT_COHORT_COUNT'],data[policy+'/counts/'+method+'__AT_COHORT_COUNT'])
            for count in ('SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400'):
                comparisons[policy+'/COUNT_'+count]=compare_official(data[policy+'/counts/COHORT__AT_'+count+'_COUNT'],data[policy+'/counts/'+count+'__AT_'+count+'_COUNT'])
        for other in ('FIELD_STATIC','COVER','UNIFORM'):
            for method in ('COHORT','SOFT_KD','HARD_KD'):
                comparisons['FIELD_vs_'+other+'/'+method]=compare_official(data['FIELD/'+method],data[other+'/'+method])
        evidence=json.loads((OUT/'field_evidence'/pair/'completion.json').read_text());assert evidence['GT_open_attempts']==0
        results[pair]=dict(HOTA={k:v['HOTA'] for k,v in data.items()},comparisons=comparisons,logical_calls=budgets,
            actual_experimental_evidence=evidence,all_development_data=True,no_target_GT_in_adaptation=True,
            prepayment_and_copied_teacher_cache_not_live_deployment_timing=True,
            descriptive_intervals_not_corrected_for_research_adaptivity=True)
        print(json.dumps(dict(event='GRAPH_FIELD_SEARCH_RESULT',pair=pair,HOTA={k:v['HOTA'] for k,v in data.items() if '/counts/' not in k},
            feedback_only=comparisons['FIELD_vs_FIELD_STATIC/COHORT'],
            same_answers_classic_vs_KD=comparisons['FIELD/FULL_vs_SOFT_KD'])),flush=True)
    atomic_json(OUT/'analysis/graph_field_search.json',results)
