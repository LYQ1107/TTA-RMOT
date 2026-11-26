"""Audit named, already observed generation receipts; never poll or score.

This is metadata verification, not a new experiment or a prediction-payload
audit. The caller must first record an actual live-tool observation using the
existing receipt ledger. No directory discovery, GT, checkpoint, prediction
values, teacher calls, CUDA operations, or numerical-source edits occur here.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_nas_namespace_resume import validate
from scripts.a100_fullscope_visual_six import OUT, PLAN, PUBLIC, METHODS, PRIMARY, COMPARATORS

SUP = ROOT / 'a100_runs/visual_classical_20261121/supervisor'
NEURAL = {'SOFT_KD288', 'HARD_KD288', 'LURE_KD288', 'SSL_KD288',
          'FLAT_KD128', 'SAME_ARCH_KD128', 'UNIFORM_SOFT_KD288'}


def audit(minimum_units, observed_chunk, public_output=None):
    assert 1 <= minimum_units <= 148
    _, plan = validate()
    canonical = json.loads((SUP / 'research_goal_active_continuation.json').read_text())
    progress = canonical['current_goal_turn_progress']
    assert canonical['goal_status'] == 'ACTIVE'
    assert progress['actual_session_id'] == 86841
    assert progress['latest_actual_session_still_running'] is True
    assert progress['latest_actual_chunk'] == observed_chunk
    assert canonical['user_requires_old_new_integration_consideration']['status'] == (
        'USER_REQUESTED_INTEGRATION_DESIGN_CONSTRAINTS_NOT_IMPLEMENTED_OR_RUN')
    ledger_path = SUP / 'observed_fullscope_generation_receipts_20261126.json'
    ledger = json.loads(ledger_path.read_text())
    assert ledger['original_registration_sha256'] == file_sha(PLAN)
    observed = ledger['cases']
    assert minimum_units <= len(observed) == progress['known_sealed_units_at_least']
    planned = {c['pair'] + '/' + c['evaluated_video'] for c in plan['cases']}
    assert len(planned) == 148 and set(observed) <= planned
    expected_methods = set(METHODS)
    expected_methods.update(PRIMARY+'__AT_'+name+'_COUNT' for name in COMPARATORS)
    expected_methods.update(name+'__AT_'+PRIMARY+'_COUNT' for name in COMPARATORS)
    assert len(expected_methods) == 31
    registration_sha256 = file_sha(PLAN)
    manifest, private = {}, []
    steps, losses, primary_losses = [], [], []
    active_questions = uniform_questions = 0
    for key, saved in sorted(observed.items()):
        pair, video = key.split('/', 1)
        path = OUT/'units'/pair/video/'generation_complete.json'
        digest = file_sha(path)
        assert digest == saved['sha256']
        done = json.loads(path.read_text())
        assert done['GT_open_attempts'] == 0
        assert done['registration_sha256'] == registration_sha256
        assert len(done['tasks']) == 31
        assert {task['method'] for task in done['tasks']} == expected_methods
        assert all(task['pair'] == pair and task['video'] == video for task in done['tasks'])
        assert set(done['ordinary_fits']) == NEURAL
        fit_summary = {}
        for name, fit in done['ordinary_fits'].items():
            n, loss = fit['total_steps'], fit['proper_teacher_excess_CE']
            assert n >= 6000 and math.isfinite(loss) and loss <= fit['adequacy_limit'] == .005
            steps.append(n)
            losses.append(loss)
            fit_summary[name] = dict(recorded_steps=n, proper_teacher_excess_CE=loss)
        primary_loss = done['primary_teacher_KL']
        assert math.isfinite(primary_loss) and primary_loss <= .005
        primary_losses.append(primary_loss)
        active_broker, uniform_broker = done['active_broker'], done['uniform_broker']
        active = active_broker['per_query_count']
        uniform = uniform_broker['per_query_count']
        assert active == active_broker['per_query_budget'] == uniform == uniform_broker['per_query_budget']
        assert active_broker['unrequested_replies_exposed_to_learner'] == 0
        assert uniform_broker['unrequested_replies_exposed_to_learner'] == 0
        active_questions += sum(active.values())
        uniform_questions += sum(uniform.values())
        manifest[key] = digest
        private.append(dict(case=key, receipt=str(path.relative_to(ROOT)), sha256=digest,
            ordinary_fits=fit_summary, primary_teacher_KL=primary_loss,
            active_per_query_count=active, uniform_per_query_count=uniform))
    assert len(steps) == 7 * len(observed) and active_questions == uniform_questions
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    counts = Counter(v['pair'] for v in observed.values())
    planned_counts = Counter(c['pair'] for c in plan['cases'])
    result = dict(status='VERIFIED_OBSERVED_GENERATION_METADATA_NOT_CURRENT_QUALITY',
        time_unix=time.time(), actual_live_session_supplied_by_caller=86841,
        caller_observed_chunk=observed_chunk, registered_units=148,
        verified_completed_units_lower_bound=len(observed), registered_outputs=4588,
        verified_reported_outputs_lower_bound=31 * len(observed),
        by_setting={k:dict(completed=counts[k], planned=planned_counts[k])
                    for k in sorted(planned_counts)},
        verified_ordinary_fits=len(steps), minimum_recorded_steps=min(steps),
        maximum_recorded_steps=max(steps), maximum_proper_teacher_excess_CE=max(losses),
        unchanged_adequacy_limit=.005, all_recorded_primary_teacher_fit_gates_pass=True,
        maximum_recorded_primary_teacher_KL=max(primary_losses),
        all_active_uniform_per_query_budgets_equal=True,
        recorded_active_unique_question_total=active_questions,
        recorded_uniform_unique_question_total=uniform_questions,
        unrequested_answers_exposed=0, GT_open_attempts_in_generation_receipts=0,
        original_registration_sha256=file_sha(PLAN), source_pins_verified=len(plan['source_sha256']),
        audit_source_sha256=file_sha(__file__), observation_ledger_file_sha256=file_sha(ledger_path),
        complete_receipt_manifest=manifest,
        sorted_case_to_receipt_sha256_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        receipt_relative_pattern=str(OUT.relative_to(ROOT))+'/units/{pair}/{video}/generation_complete.json',
        prediction_payload_or_teacher_weight_seals_checked_by_this_audit=False,
        whole4588_and_all_teacher_attempt_seals_before_current_quality=True,
        current_study_GT_or_quality_reads_by_this_audit=0,
        checkpoint_or_prediction_values_read_by_this_audit=0,
        new_GPU_or_teacher_operations_by_this_audit=0,
        fresh_single_method_E2E_cost_proven=False,
        old_new_integration_implemented_or_run=False,
        all_existing_data_are_DEV=True, goal_status='ACTIVE', scientific_goal_met=False)
    private_path = SUP / ('fullscope_observed_integrity_'+str(len(observed))+'_private_20261126.json')
    atomic_json(private_path, dict(summary=result, cases=private,
        numerical_source_sha256=plan['source_sha256']))
    if public_output is not None:
        path = public_output.resolve()
        assert path.parent == PUBLIC.resolve() and not path.exists()
        atomic_json(path, result)
    print(json.dumps(dict(status=result['status'], verified_units=len(observed),
        reported_outputs=31*len(observed), ordinary_fits=len(steps),
        maximum_proper_teacher_excess_CE=max(losses), source_pins_verified=len(plan['source_sha256']),
        GT_or_prediction_values_read=0, private_proof=str(private_path),
        public_output=str(public_output) if public_output is not None else None)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum-units', type=int, default=100)
    parser.add_argument('--observed-chunk', required=True)
    parser.add_argument('--public-output', type=Path)
    args = parser.parse_args()
    audit(args.minimum_units, args.observed_chunk, args.public_output)
