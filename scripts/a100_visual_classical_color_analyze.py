"""Isolated, post-seal module attribution for image-cohort and visual sweep."""
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


def analyze(include_facility=False):
    groups=['colour_prefix','colour_countmatched']
    if include_facility:groups+=['facility_prefix','facility_countmatched']
    while not all((OUT/g/'official_evaluation.json').is_file() for g in groups):time.sleep(15)
    results={}
    for pair in ('C4','C1'):
        stats={};metrics={}
        for group in groups:
            rows=json.loads((OUT/group/'official_evaluation.json').read_text())['results']
            for row in rows:
                if row['pair']!=pair:continue
                key=group+'/'+row['method'];stats[key]=read_stats(row['folder']);metrics[key]=row['evaluation']['metrics']
        for method in ['COVER__TRACK_SOFT','COVER__DIRECT_NATIVE']:
            key='strict_prefix/'+method;stats[key]=read_stats(OUT/'strict_prefix'/pair/method)
        for label in ('HARD','SOFT'):
            key='batch_KD_'+label;stats[key]=read_stats(OUT/'convergence'/pair/('COVER__KD_'+label+'_4000'))
        comparisons={}
        def add(name,a,b):comparisons[name]=compare_official(stats[a],stats[b])
        policies=[('COVER','colour_prefix','colour_countmatched')]
        if include_facility:policies.append(('FACILITY','facility_prefix','facility_countmatched'))
        for policy,group,counts in policies:
            prefix=group+'/'+policy+'__';full=prefix+'COHORT'
            for other in ['RGB_KD','NO_GRAPH','PCD_COHORT','COLOR_ONLY','STRONG_PRIOR']:
                add(policy+'__COHORT_vs_'+other,full,prefix+other)
            add(policy+'__COHORT_vs_PCD_KD',full,'strict_prefix/COVER__TRACK_SOFT')
            for label in ('HARD','SOFT'):add(policy+'__DIFFERENT_SCOPE_vs_batch_'+label,full,'batch_KD_'+label)
            for other in ['KD','NO_GRAPH','PCD','COLOR_ONLY','STRONG_PRIOR']:
                add(policy+'__COUNT_FULL_vs_'+other,counts+'/FULL__AT_FULL_COUNT',counts+'/'+other+'__AT_FULL_COUNT')
            add(policy+'__COUNT_KD_vs_KD',counts+'/FULL__AT_KD_COUNT',counts+'/KD__AT_KD_COUNT')
        add('COVER_vs_UNIFORM_same_cohort','colour_prefix/COVER__COHORT','colour_prefix/UNIFORM__COHORT')
        add('colour_feature_KD_vs_PCD_KD','colour_prefix/COVER__RGB_KD','strict_prefix/COVER__TRACK_SOFT')
        if include_facility:
            for other in ['COVER','UNIFORM']:
                for method in ['COHORT','RGB_KD']:
                    add('FACILITY_vs_'+other+'__'+method,'facility_prefix/FACILITY__'+method,'colour_prefix/'+other+'__'+method)
        costs={}
        for group in ('colour_prefix','facility_prefix') if include_facility else ('colour_prefix',):
            record=json.loads((OUT/group/pair/'completion.json').read_text());assert record['GT_open_attempts']==record['forbidden_read_attempts']==0
            costs[group]={}
            for video in record['video_receipts']:
                for policy,fit in video['fits'].items():
                    item=costs[group].setdefault(policy,dict(KD_fit_seconds=0.,postprocess_all_six_variants_seconds=0.))
                    item['KD_fit_seconds']+=fit['wall_seconds'];item['postprocess_all_six_variants_seconds']+=video['postprocess_wall_seconds'][policy]
        image=json.loads((OUT/'color_features'/pair/'completion.json').read_text());assert image['GT_open_attempts']==0
        results[pair]=dict(HOTA={k:v['HOTA'] for k,v in stats.items()},metrics=metrics,comparisons=comparisons,
            conditional_component_costs=costs,image_feature_extraction_seconds=sum(r['wall_seconds'] for r in image['receipts']),
            actual_real_images=sum(r['real_images'] for r in image['receipts']),invisible_sample_parts=sum(r.get('invisible_sample_parts',0) for r in image['receipts']),
            target_GT_in_adaptation=False,seed=0,all_development_data=True,
            no_independent_blind_validation=True,intervals_descriptive_not_adjusted_for_research_adaptivity=True,
            existing_feature_and_answer_banks_not_live_deployment_timing=True)
    name='colour_and_facility' if include_facility else 'colour_cohort'
    atomic_json(OUT/'analysis'/(name+'.json'),results)
    for pair,r in results.items():
        primary='FACILITY' if include_facility else 'COVER'
        print(json.dumps(dict(event='COLOUR_MECHANISM_ANALYSIS',pair=pair,HOTA=r['HOTA'],
            primary_comparisons={k:v for k,v in r['comparisons'].items() if k in [primary+'__COHORT_vs_RGB_KD',primary+'__COHORT_vs_NO_GRAPH',primary+'__COHORT_vs_PCD_COHORT',primary+'__COUNT_FULL_vs_KD']}),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--facility',action='store_true');a=p.parse_args();analyze(a.facility)
