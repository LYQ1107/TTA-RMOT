"""Read-only diagnosis of the original v8 closing-coverage assertion.

This never writes original v8 validation/receipts, changes acceptance criteria,
or runs a model. It distinguishes actual online strata from the counterfactual
LoRA strata of a replay whose six presence arrays were explicitly frozen.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.presence.runtime import runtime_guard
from flex_uda.active.v8 import BeliefStateV8

PUBLIC = ROOT / 'results/presence_active/redesign_v8p'


def coverage_bad(belief):
    return belief['absent'] & ~np.all(belief['covered_strata'] >= belief['needed_strata'], axis=1)


def diagnose():
    opened = runtime_guard()
    manifest_path = ROOT / 'results/presence_active/redesign_v8/execution.json'
    manifest = json.loads(manifest_path.read_text())
    old_public, active = Path(manifest['public_root']), Path(manifest['active_root'])
    plan = json.loads((old_public / 'registered_jobs.json').read_text())
    summaries, issues, seals = [], [], []
    all_source_coverage = all_frozen = all_questions = all_ds = True
    snapshot_count = 0
    keys = ('frame_id', 'track_id', 'query_id', 'effective_view', 'yes_probability',
            'yes_logit', 'no_logit', 'round', 'kind')
    for job in plan['jobs']:
        folder = active / job['pair'] / job['method'] / 'rho_0.3'
        seal_path = folder / 'predictions_seal.json'
        seal = json.loads(seal_path.read_text())
        assert seal['GT_open_attempts'] == 0
        seals.append(dict(job, path=str(seal_path), sha256=file_sha(seal_path)))
        failures = 0
        for directory in sorted((folder / 'videos').iterdir()):
            with np.load(directory / 'belief.npz') as b:
                bad = np.flatnonzero(coverage_bad(b))
                failures += len(bad)
                if job['method'] != 'SLOW_REPLAY':
                    for q in bad:
                        issues.append(dict(job, video=directory.name, query=str(b['queries'][q]),
                            current_covered=b['covered_strata'][q].tolist(),
                            current_needed=b['needed_strata'][q].tolist(), source_decision=None))
                    continue
                source = active / job['pair'] / 'MINUS_SLOW/rho_0.3/videos' / directory.name
                with np.load(source / 'belief.npz') as original:
                    source_ok = not coverage_bad(original).any()
                    all_source_coverage &= source_ok
                    assert np.array_equal(b['queries'], original['queries'])
                    assert np.array_equal(b['tracks'], original['tracks'])
                    for q in bad:
                        issues.append(dict(job, video=directory.name, query=str(b['queries'][q]),
                            current_covered=b['covered_strata'][q].tolist(),
                            current_needed=b['needed_strata'][q].tolist(),
                            source_covered=original['covered_strata'][q].tolist(),
                            source_needed=original['needed_strata'][q].tolist(),
                            source_absent=bool(original['absent'][q]),
                            source_decision_coverage_ok=not bool(coverage_bad(original)[q]),
                            changed_counterfactual_track_strata=int(np.sum(
                                b['prior_strata'][q] != original['prior_strata'][q]))))
            for r in range(5):
                with np.load(source / f'presence_round{r}.npz') as a, np.load(directory / f'presence_round{r}.npz') as b:
                    all_frozen &= all(a[k].tobytes() == b[k].tobytes() for k in BeliefStateV8.snapshot_fields)
                snapshot_count += 1
            original_questions = json.loads((source / 'questions.json').read_text())
            questions = json.loads((directory / 'questions.json').read_text())
            all_questions &= len(questions) == len(original_questions) and all(
                tuple(a[k] for k in keys) == tuple(b[k] for k in keys)
                for a, b in zip(questions, original_questions))
            original_receipt = json.loads((source / 'receipt.json').read_text())
            receipt = json.loads((directory / 'receipt.json').read_text())
            all_ds &= [r['ds'] for r in receipt['rounds']] == [r['ds'] for r in original_receipt['rounds']]
        summaries.append(dict(job, current_final_coverage_mismatch_queries=failures))
    assert not opened
    online_failures = sum(r['current_final_coverage_mismatch_queries'] for r in summaries
                          if r['method'] != 'SLOW_REPLAY')
    replay_failures = sum(r['current_final_coverage_mismatch_queries'] for r in summaries
                          if r['method'] == 'SLOW_REPLAY')
    result = dict(status='DIAGNOSTIC_ONLY_ORIGINAL_AUDIT_FAILURE_NOT_OVERRIDDEN',
        original_validation_or_predictions_changed=False, original_supervisor_exit_code=1,
        online_current_coverage_failures=online_failures,
        replay_counterfactual_current_coverage_mismatches=replay_failures,
        all_source_decision_coverage_valid=bool(all_source_coverage),
        replay_snapshot_count=snapshot_count, all_six_presence_arrays_byte_exact=bool(all_frozen),
        replay_questions_and_answers_exact=bool(all_questions), replay_DS_receipts_exact=bool(all_ds),
        frozen_replay_strata_provenance_explains_failure=bool(
            online_failures == 0 and replay_failures > 0 and all_source_coverage and
            all_frozen and all_questions and all_ds),
        GT_open_attempts=0, model_or_teacher_calls=0, jobs=summaries,
        all_mismatched_queries=issues, prediction_seals=seals)
    atomic_json(PUBLIC / 'original_v8_audit_diagnostic.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('all_mismatched_queries', 'prediction_seals')}), flush=True)
    return result


if __name__ == '__main__':
    diagnose()
