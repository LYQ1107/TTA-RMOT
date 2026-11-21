"""Fixed-seed mean effects; parent videos remain the independent data units."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.analyze_pf_video_uncertainty import paired_clusters

SEEDS=(0,17,42)
BASELINES=('SOFT_KD_400','SOFT_KD_4000','SOFT_KD','HARD_KD_400','HARD_KD_4000','HARD_KD')


def paired(a,b):
    parents=a[0]['parent_videos'];n=len(parents)
    assert all(s['parent_videos']==parents for s in a+b)
    counts=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=10000)
    draws=np.stack([paired_clusters(x['parent_stats'],y['parent_stats'],counts) for x,y in zip(a,b)],axis=1)
    seed_counts=np.random.default_rng(1).multinomial(len(a),np.ones(len(a))/len(a),size=10000)
    mean_draw=draws.mean(1);hierarchical=(draws*seed_counts).sum(1)/len(a)
    points=[x['HOTA']-y['HOTA'] for x,y in zip(a,b)]
    return dict(mean_delta_HOTA=float(np.mean(points)),per_seed_delta_HOTA=points,
        original_parent_clusters=n,fixed_seed_parent_CI95=np.percentile(mean_draw,[2.5,97.5]).tolist(),
        parent_and_seed_resampling_sensitivity_CI95=np.percentile(hierarchical,[2.5,97.5]).tolist(),
        bootstrap_samples=10000,bootstrap_seed=0,seed_resampling_seed=1,
        seeds_are_algorithmic_repeats_not_independent_target_datasets=True)


def main(group_suffix='',counts_suffix='_counts',output_name='three_seed_results',c1_exclusive=False):
    while not all((OUT/('strong_cover_seed'+str(s)+group_suffix+suffix)/'official_evaluation.json').is_file() for s in SEEDS for suffix in ('',counts_suffix)):time.sleep(15)
    if c1_exclusive:
        while not all((OUT/('strong_cover_seed'+str(s)+group_suffix+'_counts_exclusive')/'official_evaluation.json').is_file() for s in SEEDS):time.sleep(15)
    results={};allstats={}
    for pair in ('C4','C1'):
        data={};hota={}
        for seed in SEEDS:
            group='strong_cover_seed'+str(seed)+group_suffix;data[seed]={}
            count_family=group+('_counts_exclusive' if c1_exclusive and pair=='C1' else counts_suffix)
            for family in (group,count_family):
                record=json.loads((OUT/family/pair/'completion.json').read_text())
                assert record['GT_open_attempts']==0
                for row in record['methods']:
                    key=('counts/' if family==count_family else '')+row['method']
                    data[seed][key]=read_stats(OUT/family/pair/row['method'])
            completion=json.loads((OUT/group/pair/'completion.json').read_text())
            assert completion['GT_open_attempts']==completion['forbidden_read_attempts']==0
            assert completion.get('acquisition_seed',0)==seed
            hota[str(seed)]={k:v['HOTA'] for k,v in data[seed].items()}
        comparisons={};primary=[data[s]['COHORT'] for s in SEEDS];full_counts=[data[s]['counts/COHORT__AT_COHORT_COUNT'] for s in SEEDS]
        for reference in BASELINES+('NO_GRAPH','STRONG_PRIOR','WHITENED_COHORT','PEER','UNIFORM_PRECISION'):
            comparisons['FULL_vs_'+reference]=paired(primary,[data[s][reference] for s in SEEDS])
            comparisons['COUNT_FULL_vs_'+reference]=paired(full_counts,[data[s]['counts/'+reference+'__AT_COHORT_COUNT'] for s in SEEDS])
        means={key:float(np.mean([data[s][key]['HOTA'] for s in SEEDS])) for key in data[0]}
        # This post-hoc envelope is explicitly a diagnostic comparison bar,
        # never a target-GT checkpoint selector deployed inside the algorithm.
        strongest=max(BASELINES,key=lambda k:means[k])
        envelope=[max(data[s][k]['HOTA'] for k in BASELINES) for s in SEEDS]
        results[pair]=dict(per_seed_HOTA=hota,mean_HOTA=means,comparisons=comparisons,
            strongest_observed_fixed_KD_checkpoint=strongest,
            primary_vs_strongest_fixed_checkpoint=comparisons['FULL_vs_'+strongest],
            primary_minus_per_seed_KD_envelope=[data[s]['COHORT']['HOTA']-v for s,v in zip(SEEDS,envelope)],
            strongest_and_envelope_are_evaluation_diagnostics_not_method_inputs=True,
            seeds=list(SEEDS),all_development_data=True,no_target_GT_in_adaptation=True,
            C1_exclusivity_preserved_in_CountMatched=c1_exclusive or counts_suffix=='_counts_exclusive',
            no_unseen_target_confirmation=True,descriptive_intervals_not_corrected_for_research_adaptivity=True)
        allstats[pair]={str(s):{k:dict(parents=v['parent_videos'],numerator=v['parent_stats'][0].tolist(),
            denominator=v['parent_stats'][1].tolist()) for k,v in data[s].items()} for s in SEEDS}
        print(json.dumps(dict(event='THREE_SEED_MECHANISM_RESULT',pair=pair,mean_HOTA={k:v for k,v in means.items() if not k.startswith('counts/')},
            best_KD=strongest,comparison=results[pair]['primary_vs_strongest_fixed_checkpoint'],
            count_matched_vs_soft=comparisons['COUNT_FULL_vs_SOFT_KD'])),flush=True)
    atomic_json(OUT/'analysis'/(output_name+'.json'),results)
    atomic_json(OUT/'analysis'/(output_name.replace('_results','_sufficient_statistics')+'.json'),allstats)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group-suffix',default='');p.add_argument('--counts-suffix',default='_counts');p.add_argument('--output-name',default='three_seed_results');p.add_argument('--c1-exclusive',action='store_true')
    a=p.parse_args();main(a.group_suffix,a.counts_suffix,a.output_name,a.c1_exclusive)
