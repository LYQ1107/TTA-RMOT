"""Post-seal statistics for the causal-prefix mechanisms and exact controls."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_mechanism_analyze import compare_official


def analyze(wait=False):
    required=[OUT/'strict_prefix/priority_official_evaluation.json',OUT/'strict_prefix/additional_baselines.json']
    required += [OUT/g/'official_evaluation.json' for g in ('exemplar_prefix','continuity_prefix','prefix_countmatched','frame_convergence')]
    while not all(p.is_file() for p in required):
        if not wait:raise FileNotFoundError([str(p) for p in required if not p.is_file()])
        time.sleep(15)
    results={}
    for pair in ('C4','C1'):
        stats={};metrics={}
        for group in ('exemplar_prefix','continuity_prefix','prefix_countmatched','frame_convergence'):
            r=json.loads((OUT/group/'official_evaluation.json').read_text())
            for row in r['results']:
                if row['pair']!=pair:continue
                key=group+'/'+row['method'];stats[key]=read_stats(row['folder']);metrics[key]=row['evaluation']['metrics']
        for row in json.loads(required[0].read_text())['results']:
            if row['pair']!=pair:continue
            key='strict_prefix/'+row['method'];stats[key]=read_stats(row['folder']);metrics[key]=row['evaluation']['metrics']
        extra=OUT/'strict_prefix/additional_baselines.json'
        if extra.is_file():
            for row in json.loads(extra.read_text())['results']:
                if row['pair']!=pair:continue
                key='strict_prefix/'+row['method'];stats[key]=read_stats(row['folder']);metrics[key]=row['evaluation']['metrics']
        for label in ('HARD','SOFT'):
            key='convergence/COVER__KD_'+label+'_4000';stats[key]=read_stats(OUT/key.replace('convergence/','convergence/'+pair+'/',1))
        full='exemplar_prefix/COVER__SUPPORT_MIX';comparisons={}
        def compare(name,a,b):
            if a in stats and b in stats:comparisons[name]=compare_official(stats[a],stats[b])
        for method in ('TRACK_SOFT','FRAME_SOFT','FRAME_STRUCTURED','ENSEMBLE','DIRECT_NATIVE'):
            compare('FULL_vs_'+method,full,'strict_prefix/COVER__'+method)
        for method in ('KERNEL','MEAN','SUPPORT_GATED'):
            compare('FULL_vs_'+method,full,'exemplar_prefix/COVER__'+method)
        compare('COVER_vs_UNIFORM',full,'exemplar_prefix/UNIFORM__SUPPORT_MIX')
        compare('optional_graph_increment','exemplar_prefix/COVER__SUPPORT_MIX_GRAPH',full)
        compare('strict_HMM_primary_vs_KD','strict_prefix/COVER__GRAPH_MEMORY_HMM','strict_prefix/COVER__TRACK_SOFT')
        for source,reference in [('KD','strict_prefix/COVER__TRACK_SOFT'),('EXEMPLAR',full)]:
            compare(source+'_continuity_vs_original','continuity_prefix/'+source+'__VISUAL_SEMANTIC',reference)
            compare(source+'_semantic_continuity_increment','continuity_prefix/'+source+'__VISUAL_SEMANTIC','continuity_prefix/'+source+'__VISUAL')
        compare('FULL_vs_KD_same_associations','continuity_prefix/EXEMPLAR__VISUAL_SEMANTIC','continuity_prefix/KD__VISUAL_SEMANTIC')
        for rank in ('KD_TRACK','KD_FRAME','KD_STRUCT','ENSEMBLE','KERNEL','MEAN','NO_MEMORY'):
            compare('COUNT_FULL__FULL_vs_'+rank,'prefix_countmatched/FULL__AT_FULL_COUNT','prefix_countmatched/'+rank+'__AT_FULL_COUNT')
        for count in ('KD_TRACK','KD_STRUCT'):
            compare('COUNT_'+count+'__FULL_vs_'+count,'prefix_countmatched/FULL__AT_'+count+'_COUNT','prefix_countmatched/'+count+'__AT_'+count+'_COUNT')
        for label in ('HARD','SOFT'):
            compare('DIFFERENT_SCOPE__FULL_vs_batch_track_'+label,full,'convergence/COVER__KD_'+label+'_4000')
        for frame in ('HARD','SOFT','HARD_4000_STRUCTURED','SOFT_4000_STRUCTURED'):
            method='COVER__FRAME_KD_'+frame+('' if frame.endswith('STRUCTURED') else '_4000')
            compare('DIFFERENT_SCOPE__FULL_vs_batch_'+method,full,'frame_convergence/'+method)
        seals={}
        for group in ('strict_prefix','exemplar_prefix','continuity_prefix','prefix_countmatched'):
            r=json.loads((OUT/group/pair/'completion.json').read_text());assert r['GT_open_attempts']==0
            if 'forbidden_read_attempts' in r:assert r['forbidden_read_attempts']==0
            seals[group]=dict(status=r['status'],methods=len(r['methods']),GT_open_attempts=0)
        gp=json.loads((OUT/'exemplar_prefix'/pair/'completion.json').read_text())
        strict=json.loads((OUT/'strict_prefix'/pair/'completion.json').read_text())
        costs={p:dict(kernel_fit_predict_seconds=sum(v['fits'][p]['fit_and_predict_wall_seconds'] for v in gp['video_receipts']),
            track_KD_fit_seconds=sum(v['fits'][p]['track']['wall_seconds'] for v in strict['video_receipts']),
            frame_KD_fit_seconds=sum(v['fits'][p]['frame']['wall_seconds'] for v in strict['video_receipts'])) for p in ('COVER','UNIFORM')}
        continuation=json.loads((OUT/'continuity_prefix'/pair/'completion.json').read_text())
        links={name:sum(len(r['links'][name]['links']) for r in continuation['receipts']) for name in ('VISUAL','VISUAL_SEMANTIC')}
        results[pair]=dict(HOTA={k:v['HOTA'] for k,v in stats.items()},metrics=metrics,comparisons=comparisons,
            isolation=seals,conditional_component_wall_seconds=costs,continuity_links=links,
            continuity_endpoint_and_matching_seconds=sum(r['wall_seconds'] for r in continuation['receipts']),
            all_development_data=True,seed=0,not_an_independent_blind_test=True,
            batch_rows_are_different_information_scope=True,
            costs_not_live_end_to_end=True,confidence_intervals_descriptive_not_corrected_for_research_adaptivity=True)
    atomic_json(OUT/'analysis/causal_prefix_results.json',results)
    summary={p:dict(HOTA={k:v for k,v in r['HOTA'].items() if k.startswith('exemplar_prefix/COVER') or k.startswith('continuity_prefix/')},
        comparisons={k:v for k,v in r['comparisons'].items() if k in ('FULL_vs_TRACK_SOFT','FULL_vs_MEAN','FULL_vs_SUPPORT_GATED','COVER_vs_UNIFORM','COUNT_FULL__FULL_vs_KD_TRACK','FULL_vs_KD_same_associations')}) for p,r in results.items()}
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();analyze(a.wait)
