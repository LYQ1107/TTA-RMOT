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


def reproduce(directory,stem):
    filename=stem+'_parent_statistics.json';payload=json.loads((directory/filename).read_text());cache={}
    stats={name:resolve_record(directory,filename,name,cache) for name in payload}
    report=json.loads((directory/(stem+'_analysis.json')).read_text())
    contrasts={'transfer':TRANSFER,'query_consistency':QUERY}.get(stem)
    if contrasts is None:
        registration=json.loads((directory/(stem+'_registration_v1.json')).read_text());contrasts=registration['contrasts']
    maximum=0.;effects=0;means=0
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
    result=dict(status='PASS_SIX_SETTING_'+stem.upper()+'_EXACT_STATISTICS_REPRODUCED',HOTA_records=len(stats),
        repeated_mean_records=means,paired_effect_records=effects,maximum_absolute_error=maximum,
        bootstrap_samples=10000,bootstrap_seed=0,neural_seeds=list(SEEDS),
        original_parent_not_clip_resampling=True,KITTI_four_parent_CI_explicitly_not_claimed=True,
        no_raw_assets_or_GPU_or_Torch_required=True,arithmetic_reproduction_is_not_independent_validation=True)
    (directory/(stem+'_statistical_reproduction.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stem',default='transfer');parser.add_argument('--directory',type=Path,default=ROOT/'docs/a100_six_direction_20261122');args=parser.parse_args()
    reproduce(args.directory,args.stem)
