"""Predeclared query-only strata, evaluated only after each study's own seal.

This is descriptive mechanism diagnosis on development data. No predictions,
training, selection, thresholds or checkpoint decisions are modified here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT as SIX,PUBLIC,PAIRS,SEEDS

OUT=SIX/'query_strata'
STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_SIX_SETTINGS_QUERY_STRATA_DIAGNOSIS_RESEARCH_GOAL_ACTIVE'
GROUPS=('ALL','APPEARANCE','SPATIAL','MOTION','ROLE','OTHER','CUE_SUPPORTED',
        'CUE_UNSUPPORTED','PARSER_SUPPORTED','PARSER_UNSUPPORTED',
        'COMPOSITIONAL_COLOUR_WITHOUT_CUE','REFERENCE_SCOPE_REVIEW')


def inventory():
    from flex_uda.a100.mechanism import generation_guard,metadata
    from scripts.a100_language_scope_audit import classify
    guard=generation_guard();targets={}
    for target in sorted({t for _,t in PAIRS.values()}):
        groups={g:[] for g in GROUPS}
        for q in metadata(target)['episodes']:
            row=classify(q['query_text']);key=q['video_id']+'+'+q['expression_id']
            selected=['ALL']+row['types']
            selected.append('CUE_SUPPORTED' if row['appearance_cue_available'] else 'CUE_UNSUPPORTED')
            selected.append('PARSER_SUPPORTED' if row['supported'] else 'PARSER_UNSUPPORTED')
            if row['compositional_colour_not_in_current_cue']:selected.append('COMPOSITIONAL_COLOUR_WITHOUT_CUE')
            if row['reference_scope_review']:selected.append('REFERENCE_SCOPE_REVIEW')
            for group in set(selected):groups[group].append(key)
        for values in groups.values():assert len(values)==len(set(values))
        targets[target]=groups
    assert not guard
    atomic_json(OUT/'query_only_inventory.json',dict(targets=targets,GT_open_attempts=0,
        classifier_source_sha256=hashlib.sha256((ROOT/'scripts/a100_language_scope_audit.py').read_bytes()).hexdigest(),
        no_predictions_or_metric_read=True,groups_overlap=True))
    print(json.dumps(dict(event='QUERY_ONLY_STRATA_FIXED',GT_open_attempts=0,
        counts={t:{g:len(v) for g,v in row.items()} for t,row in targets.items()})),flush=True)


def studies():
    from scripts.a100_six_direction_analyze import CONTRASTS as transfer
    from scripts.a100_query_consistency_analyze import CONTRASTS as query
    from scripts.a100_view_fusion_analyze import CONTRASTS as view
    direct={
        'FULL_vs_DIRECT_TEACHER_FILL':('DS_FULL','RAW_TEACHER_FILL'),
        'RANK_AT_FULL_COUNT':('DS_FULL__AT_DS_FULL_COUNT','RAW_TEACHER_FILL__AT_DS_FULL_COUNT'),
        'RANK_AT_DIRECT_FILL_COUNT':('DS_FULL__AT_RAW_TEACHER_FILL_COUNT','RAW_TEACHER_FILL__AT_RAW_TEACHER_FILL_COUNT')}
    return [dict(name='transfer',root=SIX,count=234,contrasts=transfer),
        dict(name='query_consistency',root=SIX/'query_consistency',count=252,contrasts=query),
        dict(name='view_fusion',root=SIX/'view_fusion',count=468,contrasts=view),
        dict(name='direct_fill',root=SIX/'view_fusion/direct_fill_supplement',count=90,contrasts=direct)]


def analyze(study,groups):
    from scripts.active_eval import official_stats
    from scripts.analyze_pf_video_uncertainty import HOTA,sufficient,pooled_hota
    base=study['root'];barrier=json.loads((base/'all_predictions_sealed_before_GT.json').read_text())
    assert barrier['methods']==study['count'] and barrier['GT_open_attempts']==0
    assert (base/'analysis.json').is_file()
    names=sorted({n for pair in study['contrasts'].values() for n in pair})
    metric=HOTA({'PRINT_CONFIG':False});values={};stats={};max_error=0.
    for pair,(_,target) in PAIRS.items():
        by_group={g:dict(queries=len(keys),original_videos=len({k.split('+')[0] for k in keys}),
            per_seed_HOTA={},mean_HOTA={},comparisons={}) for g,keys in groups[target].items()}
        for seed in SEEDS:
            raw=base/'quality'/pair/f'seed{seed}'
            assert (raw/'completion.json').is_file()
            for name in names:
                folder=raw/'countmatched'/name if '__AT_' in name else raw/name
                if (folder/'official_equivalent.json').is_file():
                    folder=Path(json.loads((folder/'official_equivalent.json').read_text())['identical_prediction_reference'])
                row=official_stats(folder)
                assert set(row['rows'])==set(groups[target]['ALL'])
                for group,keys in groups[target].items():
                    if not keys:continue
                    selected={k:row['rows'][k] for k in keys}
                    combined=metric.combine_sequences(selected)
                    hota=100*float(np.mean(combined['HOTA']))
                    num,den=sufficient([combined]);reconstructed=float(pooled_hota((num,den),np.ones(1)))
                    max_error=max(max_error,abs(hota-reconstructed))
                    if group=='ALL':max_error=max(max_error,abs(hota-row['HOTA_unrounded']))
                    by_group[group]['per_seed_HOTA'].setdefault(str(seed),{})[name]=hota
                    stats[pair+f'/seed{seed}/'+group+'/'+name]=dict(HOTA=hota,
                        numerator=num[0].tolist(),denominator=den[0].tolist())
        for group,row in by_group.items():
            if not row['queries']:continue
            h=row['per_seed_HOTA'];row['mean_HOTA']={n:float(np.mean([h[str(s)][n] for s in SEEDS])) for n in names}
            for contrast,(a,b) in study['contrasts'].items():
                delta=[h[str(s)][a]-h[str(s)][b] for s in SEEDS]
                row['comparisons'][contrast]=dict(delta_HOTA=float(np.mean(delta)),per_neural_seed_delta=delta)
        values[pair]=by_group
    assert max_error<1e-8,max_error
    result=dict(status='DESCRIPTIVE_QUERY_STRATA_ANALYZED',study=study['name'],pairs=values,
        own_study_sealed_outputs=study['count'],new_teacher_encoder_head_or_prediction_calls=0,
        query_only_groups_fixed_before_this_analysis=True,groups_overlap_not_additive=True,
        ALL_and_subset_HOTA_reconstruction_max_error=max_error,
        statistical_scope='Development data; points and neural-seed sensitivity only, no subgroup population CI or automatic module acceptance.',
        KITTI_original_videos=4,no_GT_for_adaptation_or_strata=True,goal_status='ACTIVE')
    atomic_json(OUT/(study['name']+'_analysis.json'),result)
    atomic_json(OUT/(study['name']+'_subset_statistics.json'),stats)
    compact={p:{g:dict(queries=r['queries'],original_videos=r['original_videos'],comparisons={n:round(v['delta_HOTA'],6)
        for n,v in r['comparisons'].items()}) for g,r in row.items()} for p,row in values.items()}
    # Public overview is rounded for reading; exact arrays and full values stay
    # local. This is never presented as the exact statistical-reproduction file.
    (PUBLIC/(study['name']+'_query_strata_overview.json')).write_text(json.dumps(dict(
        study=study['name'],descriptive_rounded_display_only=True,pairs=compact,
        groups_overlap_not_additive=True,no_population_CI=True,goal_status='ACTIVE'),separators=(',',':'))+'\n')
    return result


def main():
    inv=json.loads((OUT/'query_only_inventory.json').read_text());assert inv['GT_open_attempts']==0
    assert inv['classifier_source_sha256']==hashlib.sha256((ROOT/'scripts/a100_language_scope_audit.py').read_bytes()).hexdigest()
    plan=json.loads((PUBLIC/'query_strata_registration_v1.json').read_text())
    assert plan['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pending=studies();done=[];start=time.perf_counter()
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),
        completed_studies=done,pending=[s['name'] for s in pending],wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    while pending:
        state('AWAITING_SEALED_OFFICIAL_STUDY_ANALYSES_FOR_QUERY_STRATA')
        for study in list(pending):
            if not (study['root']/'analysis.json').is_file():continue
            dest=OUT/(study['name']+'_analysis.json')
            if not dest.is_file():analyze(study,inv['targets'])
            assert json.loads(dest.read_text())['status']=='DESCRIPTIVE_QUERY_STRATA_ANALYZED'
            done.append(study['name']);pending.remove(study)
            print(json.dumps(dict(event='QUERY_STRATA_STUDY_DONE',study=study['name'],goal_status='ACTIVE')),flush=True)
        if pending:time.sleep(30)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',action='store_true');args=parser.parse_args()
    if args.inventory:inventory()
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_QUERY_STRATA_DIAGNOSIS_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
