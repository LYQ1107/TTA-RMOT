"""Keep execution success separate from the registered scientific criteria."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_weighted_classical_analyze import ORDINARY

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def quality_checks(quality):
    checks={};details={};calibration=[]
    for pair in ('C4','C1'):
        row=quality['repeated']['PROTOCOL_CALIBRATION_'+pair]['comparisons']
        for name in ('KD','BASE_PROTOCOL','KD_MEAN','UNWEIGHTED_KD'):
            effect=row['PROTOCOL_JOINT_vs_'+name]
            checks[pair+'_vs_'+name]=all(v>0 for v in effect['per_neural_seed_delta']) and effect['parent_and_neural_seed_sensitivity_CI95'][0]>0
            details[pair+'_vs_'+name]=effect
        # Preserve every older competitive ordinary readout; a higher point
        # must not be hidden just because the canonical pair passed its CI.
        names=('KD400','HARD400','HARD_FINAL')+ORDINARY
        checks[pair+'_all_other_ordinary_points_positive']=all(row['PROTOCOL_JOINT_vs_'+name]['delta_HOTA']>0 for name in names)
        calibration.append(row['PROTOCOL_JOINT_vs_PROTOCOL_GRAPH']['parent_and_neural_seed_sensitivity_CI95'][0]>0)
    checks['PAVA_conditional_contribution_atleast_one_direction']=any(calibration)
    c1=quality['repeated']['PROTOCOL_CALIBRATION_C1']['comparisons']
    checks['CHECKLIST_global_context_conditional_on_PAVA']=c1['PROTOCOL_JOINT_vs_FINAL_CAL']['parent_and_neural_seed_sensitivity_CI95'][0]>0
    checks['frozen_FINAL_masks_unchanged']=not any(r['frozen_final_mask_differences'] for r in quality['conditions'].values())
    checks['CountSelf_unchanged']=not any(any(r['CountSelf_changed_query_files'].values()) for r in quality['conditions'].values())
    return checks,details


def main():
    quality=json.loads((PUBLIC/'protocol_calibration_analysis.json').read_text())
    reproduced=json.loads((PUBLIC/'protocol_calibration_statistical_reproduction.json').read_text())
    assert reproduced['status'].startswith('PASS_') and quality['all114_new_outputs_sealed_before_manual_GT']
    checks,details=quality_checks(quality);passed=all(checks.values())
    result=dict(status='QUALITY_ACCEPTED_ACTUAL_COST_REQUIRED' if passed else 'PRIMARY_NOT_ACCEPTED',
        candidate_finally_accepted=False,quality_gate_passed=passed,checks=checks,details=details,
        full_effects_reference='protocol_calibration_analysis.json repeated.PROTOCOL_CALIBRATION_C4/C1',
        same_final_head_and_protocol_information=True,all114_sealed_before_study_manual_GT=True,
        independence='Development-derived simplification on already used data; not blind or multiplicity-corrected confirmation',
        ranking='Both quantity projections mandatory; contextual base-rate/quantity calibration is not universal semantic ordering',
        actual_cost='Own four fresh arms required if quality accepted; never substitute earlier mean-head timings',
        independent_arithmetic_reproduction=reproduced,goal_status='ACTIVE')
    atomic_json(PUBLIC/'protocol_calibration_scientific_review.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
