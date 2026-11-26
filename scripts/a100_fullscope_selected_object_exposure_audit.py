"""Describe sealed paid-action UID coverage, without probabilities or GT."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT / 'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
PLAN_SHA = 'd159ed56af4d614edd68c991ab752f6341074752c648650cd61d336d5a68ba12'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    a = sum(r['active_objects'] for r in rows)
    u = sum(r['uniform_objects'] for r in rows)
    b = sum(r['budget'] for r in rows)
    return dict(
        queries=len(rows), paid_questions_per_policy=b,
        active_query_object_visits=a, uniform_query_object_visits=u,
        active_object_coverage_fraction_of_uniform=a / u,
        query_mean_active_object_coverage_fraction_of_uniform=float(
            np.mean([r['active_objects'] / r['uniform_objects'] for r in rows])),
        queries_active_revisited_objects=sum(r['active_objects'] < r['budget'] for r in rows),
        active_repeat_object_questions=b - a,
        uniform_repeat_object_questions=b - u,
        active_requested_anchor_frames=sum(r['active_anchor_frames'] for r in rows),
        uniform_requested_anchor_frames=sum(r['uniform_anchor_frames'] for r in rows),
        active_question_weighted_mean_selected_track_length=float(
            sum(r['active_selected_track_length_sum'] for r in rows) / b),
        uniform_question_weighted_mean_selected_track_length=float(
            sum(r['uniform_selected_track_length_sum'] for r in rows) / b))


def uids(path, n, nq):
    # JSON also contains paid probabilities; only UID and query metadata are used.
    records = json.loads(path.read_text())['draws']
    result = defaultdict(list)
    seen = set()
    for record in records:
        uid = int(record['uid'])
        assert uid not in seen and 0 <= uid < n * nq
        seen.add(uid)
        q, oi = divmod(uid, n)
        assert int(record['query']) == q
        result[q].append(oi)
    return result


def main():
    plan_path = BASE / 'registration.json'
    assert sha(plan_path) == PLAN_SHA
    plan = json.loads(plan_path.read_text())
    barrier_path = BASE / 'all4588_sealed_before_GT.json'
    barrier = json.loads(barrier_path.read_text())
    assert barrier['registration_sha256'] == PLAN_SHA
    assert len(barrier['tasks']) == 4588 and barrier['GT_open_attempts'] == 0
    assert barrier['teacher_final_weight_seals'] == 4
    ledger = json.loads((SUP / 'observed_fullscope_generation_receipts_20261126.json').read_text())
    assert len(ledger['cases']) == len(plan['cases']) == 148
    grouped = defaultdict(list)
    by_view = defaultdict(list)
    all_rows, inputs = [], []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video']
        entry = ledger['cases'][pair + '/' + video]
        receipt = ROOT / entry['receipt']
        assert sha(receipt) == entry['sha256']
        unit = receipt.parent
        obs_path, query_path = Path(case['observations']), Path(case['temporal_queries'])
        assert sha(obs_path) == case['observations_sha256']
        assert sha(query_path) == case['temporal_queries_sha256']
        obs = np.load(obs_path, mmap_mode='r', allow_pickle=False)
        doc = json.loads(query_path.read_text())
        queries = doc['queries']
        assert len(obs) == doc['observations']
        ids, columns, lengths = np.unique(obs[:, 1], return_inverse=True, return_counts=True)
        primary_path = unit / 'primary/draws.json'
        uniform_path = unit / 'uniform_draws.json'
        primary = uids(primary_path, len(obs), len(queries))
        uniform = uids(uniform_path, len(obs), len(queries))
        for q, query in enumerate(queries):
            budget = int(query['logical_budget'])
            assert len(primary[q]) == len(uniform[q]) == budget
            aa, uu = np.asarray(primary[q]), np.asarray(uniform[q])
            at, ac = np.unique(columns[aa], return_counts=True)
            ut = np.unique(columns[uu])
            assert len(ut) == min(budget, len(ids))
            row = dict(
                pair=pair, video=video, query_index=q, query_id=query['query_id'],
                requested_view=query['view'], budget=budget, available_objects=len(ids),
                active_objects=len(at), uniform_objects=len(ut),
                active_anchor_frames=len(np.unique(obs[aa, 0])),
                uniform_anchor_frames=len(np.unique(obs[uu, 0])),
                active_max_questions_on_one_object=int(ac.max()),
                active_objects_asked_at_multiple_times=int((ac > 1).sum()),
                active_selected_track_length_sum=int(lengths[columns[aa]].sum()),
                uniform_selected_track_length_sum=int(lengths[columns[uu]].sum()))
            grouped[pair].append(row)
            by_view[(pair, query['view'])].append(row)
            all_rows.append(row)
        assert sum(len(v) for v in primary.values()) == case['logical_budget']
        inputs.append(dict(
            pair=pair, video=video, receipt_sha256=entry['sha256'],
            primary_draws_sha256=sha(primary_path), uniform_draws_sha256=sha(uniform_path),
            observations_sha256=case['observations_sha256'],
            queries_sha256=case['temporal_queries_sha256']))
    assert len(all_rows) == 3506
    result = dict(
        status='COMPLETE_ALL148_SEALED_PAID_UID_OBJECT_AND_ANCHOR_EXPOSURE_NO_GT',
        setting_video_units=148, query_instances=3506,
        by_setting={p: summarize(rows) for p, rows in sorted(grouped.items())},
        by_setting_requested_view={p + '/' + v: summarize(rows)
                                   for (p, v), rows in sorted(by_view.items())},
        per_query=all_rows, inputs=inputs,
        current_study_GT_or_HOTA_reads=0, model_parameter_or_prediction_reads=0,
        teacher_probability_values_used=0, new_GPU_teacher_or_fit_calls=0,
        actual_active_object_coverage_measured=True,
        correct_target_coverage_measured=False, source_sha256=sha(Path(__file__)),
        original_registration_sha256=PLAN_SHA,
        full_before_GT_seal_sha256=sha(barrier_path),
        interpretation=[
            'These are actual paid UID choices, not the earlier equal-priority or pure-frame-uniform reference.',
            'Objects here are official candidate Track IDs, not verified physical identities; mixed or fragmented tracks are not corrected by this count.',
            'Uniform visits distinct objects before revisiting them. Greater object coverage alone is not greater correct-target coverage.',
            'An active revisit uses a different requested anchor observation. Temporal queries may benefit from such revisits.',
            'Requested views may differ from effective teacher fallback views; nominal anchor-frame counts exclude internal CLIP history.',
            'Selected track duration is not semantic relevance; short candidates may be noise or physical track fragments.',
            'No teacher probabilities or learned priorities were used. Long-track preference or revisits cannot alone establish the causal origin of a HOTA change.',
            'Keep the current study fixed and wait for all186/180 reproduction, then relate these descriptions to the original registered postseal diagnosis.'],
        numerical_experiment_changed=False, goal_achieved=False)
    out = SUP / 'fullscope_selected_object_exposure_audit_20261126.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), by_setting=result['by_setting'],
                          actual_active_object_coverage_measured=True,
                          correct_target_coverage_measured=False,
                          current_study_GT_or_HOTA_reads=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
