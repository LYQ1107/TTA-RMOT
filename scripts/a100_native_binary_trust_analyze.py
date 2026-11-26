"""Complete, post-seal quality and observer dynamics for the RSI Trust cohort."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_native_binary_trust_six_setting import OUT,PUBLIC,PRIMARY,METHODS,COMPARATORS,RECIPES,PAIRS,EXPECTED
STEM='native_binary_trust_quality_seed0_20261124'


def main():
    barrier=json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    official=json.loads((OUT/'official.json').read_text())
    assert barrier['methods']==official['methods']==EXPECTED==312 and barrier['GT_open_attempts']==0
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    from scripts.a100_visual_classical_sparse_statistics_reproduce import pack_statistic_rows
    stats={};settings={};max_error=0.
    for pair in PAIRS:
        root=OUT/pair/'seed0';raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==counts['GT_open_attempts']==0 and counts['CountSelf_identical_masks_for_all_methods']
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        assert len(paths)==52
        rows={n:read_stats(p) for n,p in paths.items()};effects={}
        for n,r in rows.items():
            a,b=r['parent_stats'];error=abs(float(pooled_hota((a,b),np.ones(len(a))))-r['HOTA']);max_error=max(max_error,error)
            stats[pair+'_seed0/'+n]=dict(parents=r['parent_videos'],HOTA=r['HOTA'],numerator=a.tolist(),denominator=b.tolist())
        for other in COMPARATORS:
            for label,a,b in [('RAW',PRIMARY,other),('AT_PRIMARY_COUNT',PRIMARY,other+'__AT_'+PRIMARY+'_COUNT'),
                              ('AT_OTHER_COUNT',PRIMARY+'__AT_'+other+'_COUNT',other)]:
                effects[label+'_vs_'+other]=dict(comparison([rows[a]],[rows[b]]),methods=[a,b])
        ordinary={};observers={p:dict(updates=0,reward_sum=0.,paid_requests=0,probe_pairs=0,effective_view_mismatches=0,max_update_KL=0.) for p in ('LEARNED','FROZEN')};changes=[]
        for receipt in raw['receipts']:
            for name,report in receipt['fits'].items():
                if RECIPES[name]['family']=='structured':continue
                for f in report['fits']:
                    assert f['teacher_fit_gate_passed'] and f['excess_risk']<=.005
                    ordinary[name]=max(ordinary.get(name,0.),f['excess_risk'])
            sets={};orders={}
            for policy in observers:
                local=root/'videos'/receipt['video']/'policies'/policy
                paid=json.loads((local/'paid.json').read_text())['paid'];seq=json.loads((local/'sequence.json').read_text())['sequence'];lookup={r['uid']:r for r in paid}
                sets[policy]=set(lookup);orders[policy]=[r['uid'] for r in seq]
                report=receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy];dest=observers[policy]
                dest['updates']+=report['actor_actual_updates'];dest['paid_requests']+=len(paid)
                dest['reward_sum']+=sum(r['credit']['reward'] for r in report['rounds'])
                dest['max_update_KL']=max(dest['max_update_KL'],*[r['actor_update']['mean_recorded_state_KL'] for r in report['rounds']])
                for stage in range(3):
                    probe=[lookup[r['uid']] for r in seq if r['role']=='probe' and r['stage']==stage];assert len(probe)%2==0
                    dest['probe_pairs']+=len(probe)//2
                    dest['effective_view_mismatches']+=sum(a['effective_view']!=b['effective_view'] for a,b in zip(probe[::2],probe[1::2]))
            changes.append(dict(video=receipt['video'],replacement_questions=len(sets['LEARNED']^sets['FROZEN'])//2,
                complete_question_order_identical=orders['LEARNED']==orders['FROZEN']))
        settings[pair]=dict(source_target=PAIRS[pair],raw_HOTA={n:rows[n]['HOTA'] for n in METHODS},
            comparisons=effects,ordinary_maximum_excess_risk=ordinary,observer_dynamics=observers,
            learned_vs_frozen_questions=changes,generation_wall_seconds=raw['wall_seconds'])
    assert len(stats)==312 and max_error<1e-8
    result=dict(status='ANALYZED_RSI_TRUST_ALL_SIX_SETTINGS',cohort='native_binary_trust_quality_20261124',
        primary=PRIMARY,neural_seeds=[0],settings=settings,official_unique_evaluations=312,raw_methods=108,CountSelf_aliases=108,
        HOTA_parent_sum_max_error=max_error,all_outputs_sealed_before_GT=True,all_data_DEVELOPMENT=True,
        independent_blind_validation=False,quality_replay_not_cold_deployment=True,
        same_method_names_in_other_cohorts_require_explicit_cohort_identifier=True,
        full_three_factor_interactions_are_not_in_this_reduced_quality_scope=True,
        teacher_information_and_question_changes_not_RMOT_truth=True,
        actor_recorded_state_KL_constraint_not_task_return_guarantee=True,
        original_726_study_unmodified=True,final_goal_not_closed_by_analysis=True)
    atomic_json(OUT/'analysis.json',result);atomic_json(OUT/'parent_statistics_readable.json',stats)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(pack_statistic_rows(stats),separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],primary_HOTA={p:v['raw_HOTA'][PRIMARY] for p,v in settings.items()})),flush=True)


if __name__=='__main__':main()
