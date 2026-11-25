"""Post-seal analysis of the unified native-binary six-setting study.

Never run during adaptation. Counts are diagnostic; all data are DEV and
only seed0 is trained. Statistics are grouped by ORIGINAL parent videos.
"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_native_binary_six_setting import OUT,PRIMARY,METHODS,COMPARATORS,PAIRS,PUBLIC

STEM='native_binary_quality_seed0_20261124'


def factorial_cells():
    return {f'{c}{r}{d}':'FULL_'+('CONDITIONAL' if c else 'LIKELIHOOD')+'_'+
        ('REFERENCE' if r else 'LOCAL')+'_'+('LEARNED' if d else 'FROZEN')+'_FINAL'
        for c in (0,1) for r in (0,1) for d in (0,1)}


def factorial_weights():
    cells=factorial_cells();result={}
    for label, axes in (('CONDITIONAL',(0,)),('REFERENCE',(1,)),('OBSERVER',(2,)),
            ('CONDITIONAL_x_REFERENCE',(0,1)),('CONDITIONAL_x_OBSERVER',(0,2)),
            ('REFERENCE_x_OBSERVER',(1,2)),('THREE_WAY',(0,1,2))):
        result[label]={name:float(np.prod([2*int(code[a])-1 for a in axes])/2**(3-len(axes)))
                       for code,name in cells.items()}
    return result


def linear_effect(rows,weights):
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    parents=rows[next(iter(weights))]['parent_videos'];n=len(parents)
    assert all(rows[k]['parent_videos']==parents for k in weights)
    estimate=float(sum(w*rows[k]['HOTA'] for k,w in weights.items()))
    result=dict(delta_HOTA=estimate,original_parent_groups=n,CI95=None,method_weights=weights)
    if n>=5:
        rng=np.random.default_rng(0);draws=rng.multinomial(n,np.ones(n)/n,size=10000)
        effects=sum(w*pooled_hota(rows[k]['parent_stats'],draws) for k,w in weights.items())
        result.update(CI95=np.percentile(effects,[2.5,97.5]).tolist(),bootstrap_samples=10000,bootstrap_seed=0)
    return result


def main():
    # Check the complete study barrier before importing statistics providers.
    barrier=json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    official=json.loads((OUT/'official.json').read_text())
    expected=6*(len(METHODS)+2*len(COMPARATORS))
    assert barrier['methods']==official['methods']==expected==726
    assert barrier['GT_open_attempts']==0
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    from scripts.a100_visual_classical_sparse_statistics_reproduce import pack_statistic_rows
    statistics={};settings={};max_error=0.
    for pair in PAIRS:
        root=OUT/pair/'seed0'
        raw=json.loads((root/'completion.json').read_text())
        counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==counts['GT_open_attempts']==0
        assert counts['CountSelf_identical_masks_for_all_methods']
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        assert len(paths)==121
        rows={name:read_stats(path) for name,path in paths.items()}
        for name,row in rows.items():
            num,den=row['parent_stats']
            error=abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA'])
            max_error=max(max_error,error)
            statistics[pair+'_seed0/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
                numerator=num.tolist(),denominator=den.tolist())
        contrasts={}
        for other in COMPARATORS:
            contrasts['RAW_vs_'+other]=(PRIMARY,other)
            contrasts['AT_PRIMARY_COUNT_vs_'+other]=(PRIMARY,other+'__AT_'+PRIMARY+'_COUNT')
            contrasts['AT_OTHER_COUNT_vs_'+other]=(PRIMARY+'__AT_'+other+'_COUNT',other)
        effects={label:dict(comparison([rows[a]],[rows[b]]),methods=[a,b]) for label,(a,b) in contrasts.items()}
        cube=list(factorial_cells().values())
        matched={n:rows[n if n==PRIMARY else n+'__AT_'+PRIMARY+'_COUNT'] for n in cube}
        cube_effects={scope:{label:linear_effect(source,weights) for label,weights in factorial_weights().items()}
                     for scope,source in (('raw',rows),('at_primary_count',matched))}
        ordinary={};observer={p:dict(actual_actor_updates=0,paid_requests=0,reward_sum=0.,fresh_probe_rounds=0) for p in ('LEARNED','FROZEN')}
        changed=[]
        for receipt in raw['receipts']:
            for name,report in receipt['fits'].items():
                if name.startswith(('FULL_','NO_IDENTITY_')):continue
                for fit in report['fits']:
                    assert fit['teacher_fit_gate_passed']
                    ordinary[name]=max(ordinary.get(name,0.),fit['excess_risk'])
            video=receipt['video'];sets={}
            for policy in observer:
                report=receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]
                observer[policy]['actual_actor_updates']+=report['actor_actual_updates']
                observer[policy]['paid_requests']+=report['teacher_budget']['unique_teacher_bundles']
                observer[policy]['reward_sum']+=sum(r['credit']['reward'] for r in report['rounds'])
                observer[policy]['fresh_probe_rounds']+=report['fresh_probe_rounds']
                paid=json.loads((root/'videos'/video/'policies'/policy/'paid.json').read_text())['paid']
                sets[policy]={r['uid'] for r in paid}
            changed.append(dict(video=video,question_set_symmetric_difference=len(sets['LEARNED']^sets['FROZEN']),
                replacement_questions=len(sets['LEARNED']^sets['FROZEN'])//2))
        settings[pair]=dict(source_target=PAIRS[pair],neural_seeds=[0],
            raw_HOTA={n:rows[n]['HOTA'] for n in METHODS},
            comparisons=effects,factorial=cube_effects,
            ordinary_maximum_teacher_excess_risk=ordinary,
            teacher_logit_KD_risk_units="squared logit; all other ordinary arms proper KL nats",
            observer_dynamics=observer,learned_vs_frozen_questions=changed,
            generation_wall_seconds=raw['wall_seconds'],
            generation_cost_scope='Recorded teacher replay and overlapping small-head fitting; not independent cold deployment.',
            CountSelf_all_raw_masks_identical=True)
    assert max_error<1e-8 and len(statistics)==726
    result=dict(status='ANALYZED_UNIFIED_NATIVE_BINARY_SIX_SETTINGS_SEED0',primary=PRIMARY,
        neural_seeds=[0],settings=settings,official_unique_evaluations=726,
        raw_methods=246,CountSelf_mask_aliases=246,HOTA_parent_sum_max_error=max_error,
        all_six_raw_and_dual_count_outputs_sealed_before_GT=True,
        all_data_DEVELOPMENT=True,independent_blind_validation=False,
        uncertainty='Descriptive paired original-parent-video bootstrap of fixed seed0 outputs; KITTI four-parent cases have no population CI. No development/multiple-comparison correction.',
        common_count_factorial_is_diagnostic_not_unique_causal_quantity_decomposition=True,
        teacher_probabilities_are_not_semantic_truth=True,
        factorized_binary_products_are_modelling_assumptions_not_measured_joint_answers=True,
        observer_factor_uses_fixed_paths_from_two_full_solver_loops=True,
        representation_and_objective_replays_match_all_three_training_rounds=True,
        quality_replay_not_fresh_cold_cost=True,
        primary_fixed_before_GT_no_posthoc_ablation_promotion=True,
        final_scientific_goal_not_closed_by_this_analysis=True)
    atomic_json(OUT/'analysis.json',result)
    atomic_json(OUT/'parent_statistics_readable.json',statistics)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(pack_statistic_rows(statistics),separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],primary_HOTA={p:r['raw_HOTA'][PRIMARY] for p,r in settings.items()},maximum_statistic_error=max_error)),flush=True)


if __name__=='__main__':
    main()
