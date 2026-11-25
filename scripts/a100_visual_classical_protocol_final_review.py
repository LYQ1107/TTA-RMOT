"""Post-seal evidence acceptance; Goal completion remains a root-agent action."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_protocol_calibration_review import quality_checks

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def main():
    quality=json.loads((PUBLIC/'protocol_calibration_analysis.json').read_text())
    attribution=json.loads((PUBLIC/'pure_components_analysis.json').read_text())
    actual=json.loads((PUBLIC/'protocol_cold_analysis.json').read_text())
    checks,details=quality_checks(quality)
    checks.update(attribution['attribution_checks'])
    for stem in ('protocol_calibration','pure_components','protocol_cold'):
        r=json.loads((PUBLIC/(stem+'_statistical_reproduction.json')).read_text())
        checks[stem+'_independent_arithmetic']=r['status'].startswith('PASS_') and r['maximum_absolute_error']<1e-8
    checks['four_real_independent_workers']=len(actual['runs'])==4 and all(
        r['worker']['exit_code']==0 and r['worker']['independent_single_arm_processes'] and r['worker']['all_startup_and_inference_included']
        for r in actual['runs'].values())
    checks['only_GPU0_1_2_3']={r['worker']['physical_GPU'] for r in actual['runs'].values()}=={0,1,2,3}
    checks['real_teacher_QA4526_zero_cache']=actual['total_fresh_actual_QA']==4526 and all(r['teacher']['cache_hits']==0 for r in actual['runs'].values())
    checks['all_actual_outputs_sealed_before_GT']=actual['all_four_fresh_predictions_sealed_before_followup_GT']
    checks['actual_evidence_matches_both_arms_and_quality']=all(r['exact_source_questions_and_teacher_answers_equal'] for r in
        list(actual['actual_within_pair_evidence_audit'].values())+list(actual['quality_source_evidence_audit'].values()))
    checks['actual_saved_probabilities_and_query_files_exact']=not any(r['changed_query_prediction_files'] or r['maximum_saved_score_probability_error'] for r in actual['expected_seed0_prediction_audit'].values())
    checks['same_final_head_last_video_weights_and_normalization']=not any(r['last_video_head_weight_max_abs_difference'] or r['last_video_head_statistic_max_abs_difference'] for r in actual['actual_within_pair_evidence_audit'].values())
    checks['all_actual_fits_meet_teacher_KL_stop']=not any(r['teacher_fit_audit']['videos_exceeding_registered_KL_tolerance'] for r in actual['runs'].values())
    checks['actual_single_soft_label_FINAL']=all(r['protocol']['actual_trained_label_types']==['CLIP'] and r['protocol']['shared_head_readout']=='CONVERGED_FINAL' for r in actual['runs'].values())
    checks['real_individual_wall_times_reported']=all(r['worker']['actual_worker_process_wall_seconds']>0 for r in actual['runs'].values())
    result=dict(status='EVIDENCE_ACCEPTED_ON_EXISTING_DEV_DIRECTIONS' if all(checks.values()) else 'EVIDENCE_NOT_ACCEPTED',
        checks=checks,all_evidence_checks_passed=all(checks.values()),
        quality_HOTA={p:r['mean_HOTA'] for p,r in quality['repeated'].items()},
        actual_seed0_HOTA={p:r['raw_HOTA'] for p,r in actual['conditions'].items()},
        actual_wall_seconds={p:r['worker']['actual_worker_process_wall_seconds'] for p,r in actual['runs'].items()},
        actual_real_QA=actual['total_fresh_actual_QA'],
        scope=dict(official_fixed_candidates_and_IDs=True,adaptation_manual_target_GT=False,frozen_source_and_teacher=True,
            allowed_shared_small_teacher_head=True,current_video_transductive_not_frame_online=True,
            existing_data_all_development=True,no_independent_unseen_domain_or_protocol_validation=True,
            given_task_protocol_and_dataset_confounded=True),
        limitations=['C1 benefits largely quantity/base-rate calibration; both fixed-count ordering intervals cross zero.',
            'C4 query retrieval is not uniformly rescued; no claim all coverage failures resolved.',
            'Not every module has a positive conditional interval in every domain.',
            'Intervals describe fixed predictions after repeated development, not multiplicity-corrected or pipeline-refit generalization.',
            'Cold worker includes research integrity/output bookkeeping; OS/NAS caches unflushed, shared host load/staggered starts. No all-budget Pareto/speedup claim.',
            'Only referring adaptation is studied; official front-end generation and its upstream supervision are outside the cost/source-only claim.',
            'Legacy all-label-types boolean does not mean actual RAW/HARD fits; authoritative actual type is CLIP, as separately disclosed.'],
        full_sources=['protocol_calibration_analysis.json','pure_components_analysis.json','protocol_cold_analysis.json'],
        publication_and_goal_tool_checked_separately=True,
        goal_update_policy='Root explicitly updates the Goal after final publication; this checker never changes it')
    atomic_json(PUBLIC/'protocol_final_scientific_review.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,goal_auto_closed=False)),flush=True)
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
