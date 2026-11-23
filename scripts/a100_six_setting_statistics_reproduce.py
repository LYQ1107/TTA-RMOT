"""Recompute sealed six-setting HOTA/effects from public aggregates only.

NumPy and the adjacent aggregate decoder suffice; raw predictions, GT, models,
Torch and GPUs are unnecessary. Reproduction does not make DEV data a holdout.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,pooled,recompute

SEEDS=(0,17,42)
TRANSFER={
    **{'JOINT_vs_'+b:('JOINT',b) for b in ('BASE_PROTOCOL','KD_FINAL','KD_MEAN','HARD_FINAL','RAW_TEACHER_FILL','LOCAL_JOINT')},
    'CONTEXT_CONDITIONAL':('JOINT','VISUAL_ONLY'),'VISUAL_CONDITIONAL':('JOINT','PROTOCOL_GRAPH'),
    'VISUAL_ONLY_vs_KD_FINAL':('VISUAL_ONLY','KD_FINAL'),
    'RANK_AT_JOINT_COUNT':('JOINT__AT_JOINT_COUNT','BASE_PROTOCOL__AT_JOINT_COUNT'),
    'RANK_AT_BASE_COUNT':('JOINT__AT_BASE_PROTOCOL_COUNT','BASE_PROTOCOL__AT_BASE_PROTOCOL_COUNT')}
QUERY={
    **{'FULL_vs_'+b:('LATTICE_JOINT',b) for b in ('KD_FINAL','KD_MEAN','HARD_FINAL','BASE_PROTOCOL','RAW_TEACHER_FILL')},
    'CONSISTENCY_CONDITIONAL':('LATTICE_JOINT','UNPROJECTED_JOINT'),
    'CONSISTENCY_ON_KD':('LATTICE_KD','KD_FINAL'),
    'PROTOCOL_VISUAL_CONDITIONAL':('LATTICE_JOINT','LATTICE_KD'),
    'EQUIVALENCE_CONDITIONAL':('EQUIV_JOINT','UNPROJECTED_JOINT'),
    'IMPLICATION_AFTER_EQUIVALENCE':('LATTICE_JOINT','EQUIV_JOINT'),
    'TRUE_GRAPH_vs_QUERY_SHUFFLE':('LATTICE_JOINT','SHUFFLED_JOINT'),
    'RANK_AT_FULL_COUNT':('LATTICE_JOINT__AT_LATTICE_JOINT_COUNT','BASE_PROTOCOL__AT_LATTICE_JOINT_COUNT'),
    'RANK_AT_BASE_COUNT':('LATTICE_JOINT__AT_BASE_PROTOCOL_COUNT','BASE_PROTOCOL__AT_BASE_PROTOCOL_COUNT')}
CONDITIONAL_INTERACTION=(('FULL_FINAL',1),('CONDITIONAL_BALANCED_FINAL',-1),
    ('DIRECT_PROFILE_FINAL',-1),('DIRECT_BALANCED_FINAL',1))


def registered_contrasts(directory,stem):
    if stem=='transfer':return TRANSFER
    if stem=='query_consistency':return QUERY
    full='DS_FULL'
    if stem=='view_fusion':
        bases=('RAW_FINAL','RAW_MEAN','HARD_FINAL','RAW_SRC_FINAL')
        result={**{'FULL_vs_'+b:(full,b) for b in bases},
            'LEARNED_vs_FIXED_RELIABILITY':(full,'FIXED_FULL'),
            'LEARNED_vs_RAW_PROBABILITY':(full,'RAW_FULL'),
            'DS_LABELS_ON_SHARED_HEAD':('DS_FINAL','RAW_FINAL'),
            'FIXED_LABELS_ON_SHARED_HEAD':('FIXED_FINAL','RAW_FINAL'),
            'STRUCTURE_ON_DS_HEAD':(full,'DS_FINAL'),
            'CONTEXT_CONDITIONAL':('DS_JOINT','DS_VISUAL'),
            'VISUAL_CONDITIONAL':('DS_JOINT','DS_CONTEXT'),
            'QUERY_CONSISTENCY_CONDITIONAL':(full,'DS_JOINT'),
            'TRUE_QUERY_GRAPH_vs_SHUFFLE':(full,'DS_SHUFFLED')}
    elif stem in ('view_fusion_direct_fill','view_fusion_flat_rgb'):
        direct=stem=='view_fusion_direct_fill';base='RAW_TEACHER_FILL' if direct else 'RAW_RGB_FINAL'
        return {('FULL_vs_DIRECT_TEACHER_FILL' if direct else 'FULL_vs_FLAT_RGB'):(full,base),
            'RANK_AT_FULL_COUNT':(full+'__AT_'+full+'_COUNT',base+'__AT_'+full+'_COUNT'),
            ('RANK_AT_DIRECT_FILL_COUNT' if direct else 'RANK_AT_RGB_COUNT'):(full+'__AT_'+base+'_COUNT',base+'__AT_'+base+'_COUNT')}
    elif stem=='lame_reference':
        bases=('RAW_LAME_KNN5','DS_RGB_LAME_KNN5')
        result={'FULL_vs_ORDINARY_KD_LAME':(full,bases[0]),
            'FULL_vs_SAME_DS_HEAD_RGB_LAME':(full,bases[1]),
            'LAME_ON_ORDINARY_HEAD':(bases[0],'RAW_FINAL'),
            'RGB_LAME_ON_DS_HEAD':(bases[1],'DS_FINAL')}
    elif stem=='object_reference':
        return json.loads((directory/'object_reference_quality_registration_v1.json').read_text())['contrasts']
    else:
        return json.loads((directory/(stem+'_registration_v1.json')).read_text())['contrasts']
    for base in bases:
        result['RANK_AT_FULL_COUNT_vs_'+base]=(full+'__AT_'+full+'_COUNT',base+'__AT_'+full+'_COUNT')
        result['RANK_AT_BASE_COUNT_vs_'+base]=(full+'__AT_'+base+'_COUNT',base+'__AT_'+base+'_COUNT')
    return result


def reproduce(directory,stem):
    filename=stem+'_parent_statistics.json';payload=json.loads((directory/filename).read_text());cache={}
    stats={name:resolve_record(directory,filename,name,cache) for name in payload}
    report=json.loads((directory/(stem+'_analysis.json')).read_text())
    contrasts=registered_contrasts(directory,stem)
    maximum=0.;effects=0;means=0;interactions=0
    def check(actual,expected,name):
        nonlocal maximum
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        assert error<1e-8,(name,error);maximum=max(maximum,error)
    for name,row in stats.items():check(pooled(row,np.ones(len(row['parents']))),row['HOTA'],name)
    for slug,row in report.get('conditions',{}).items():
        for method,value in row['raw_HOTA'].items():check(stats[slug+'/'+method]['HOTA'],value,slug+'/'+method)
    for pair,row in report['repeated'].items():
        keys=[pair+f'_seed{s}' for s in SEEDS]
        for method,value in row['mean_HOTA'].items():
            check(np.mean([stats[k+'/'+method]['HOTA'] for k in keys]),value,pair+'/'+method);means+=1
        for label,published in row['comparisons'].items():
            a,b=contrasts[label];left=[stats[k+'/'+a] for k in keys];right=[stats[k+'/'+b] for k in keys]
            parents=left[0]['parents'];assert all(x['parents']==y['parents']==parents for x,y in zip(left,right))
            delta=[pooled(x,np.ones(len(parents)))-pooled(y,np.ones(len(parents))) for x,y in zip(left,right)]
            check(delta,published['per_neural_seed_delta'],pair+'/'+label+'/per_seed')
            check(np.mean(delta),published['delta_HOTA'],pair+'/'+label+'/mean')
            assert published['original_parent_groups']==len(parents)
            if len(parents)<5:
                assert published['CI95'] is None
                assert 'parent_and_neural_seed_sensitivity_CI95' not in published
            else:
                computed=recompute(left,right)
                for field,value in computed.items():check(value,published[field],pair+'/'+label+'/'+field)
            effects+=1
        if stem in ('conditional_observation','conditional_deterministic_repair'):
            # This is the pre-registered descriptive factorial interaction,
            # independently pooled from the public parent aggregates. Valid
            # ordinary contrasts do not by themselves verify this block.
            published=row.get('factorization_profile_interaction')
            assert published is not None,(pair,'missing registered four-arm interaction')
            parents=stats[keys[0]+'/FULL_FINAL']['parents']
            values=[]
            for key in keys:
                rows=[stats[key+'/'+method] for method,_ in CONDITIONAL_INTERACTION]
                assert all(r['parents']==parents for r in rows),(pair,'interaction parent groups differ')
                values.append(sum(coefficient*pooled(r,np.ones(len(parents)))
                    for r,(_,coefficient) in zip(rows,CONDITIONAL_INTERACTION)))
            check(values,published['per_seed'],pair+'/four_arm_interaction/per_seed')
            check(np.mean(values),published['mean'],pair+'/four_arm_interaction/mean')
            assert published.get('CI95','missing') is None,(pair,'unregistered interaction CI')
            interactions+=1
    result=dict(status='PASS_SIX_SETTING_'+stem.upper()+'_EXACT_STATISTICS_REPRODUCED',HOTA_records=len(stats),
        repeated_mean_records=means,paired_effect_records=effects,maximum_absolute_error=maximum,
        four_arm_interaction_records=interactions,four_arm_interaction_seed_values=interactions*len(SEEDS),
        bootstrap_samples=10000,bootstrap_seed=0,neural_seeds=list(SEEDS),
        original_parent_not_clip_resampling=True,KITTI_four_parent_CI_explicitly_not_claimed=True,
        no_raw_assets_or_GPU_or_Torch_required=True,arithmetic_reproduction_is_not_independent_validation=True)
    (directory/(stem+'_statistical_reproduction.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stem',default='transfer');parser.add_argument('--directory',type=Path,default=ROOT/'docs/a100_six_direction_20261122');args=parser.parse_args()
    reproduce(args.directory,args.stem)
