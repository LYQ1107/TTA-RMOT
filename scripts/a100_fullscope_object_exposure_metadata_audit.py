"""Metadata-only exposure analysis; never opens labels, models or answers."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom


ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT / 'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
PLAN = BASE / 'registration.json'
RUNTIME = ROOT / 'flex_uda/active/fullscope_visual_runtime.py'
PLAN_SHA = 'd159ed56af4d614edd68c991ab752f6341074752c648650cd61d336d5a68ba12'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                    np.quantile(a, [0, .25, .5, .75, 1]).tolist()))


def main():
    assert sha(PLAN) == PLAN_SHA
    plan = json.loads(PLAN.read_text())
    source_sha = sha(RUNTIME)
    expected = [v for k, v in plan['source_sha256'].items()
                if k.endswith('flex_uda/active/fullscope_visual_runtime.py')]
    assert expected == [source_sha]
    tree = ast.parse(RUNTIME.read_text())
    proposal_node = next(n for n in tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == 'proposal')
    # Execute only the registered numpy proposal function, with no module imports.
    namespace = {'np': np}
    exec(compile(ast.Module(body=[proposal_node], type_ignores=[]),
                 str(RUNTIME), 'exec'), namespace)
    proposal = namespace['proposal']

    columns = np.r_[np.zeros(1000, dtype=int), np.ones(10, dtype=int)]
    priority = np.ones(len(columns))
    known = np.zeros(len(columns), dtype=bool)
    first = float(proposal(priority, known, columns)[columns == 1].sum())
    known[0] = True
    after_long = float(proposal(priority, known, columns)[columns == 1].sum())
    second_or_first = first + (1 - first) * after_long
    assert abs(first - 10 / 1010) < 1e-12
    toy = dict(track_lengths=[1000, 10], questions=2,
               fixed_equal_priority_assumption=True,
               current_proposal_first_question_short_track_probability=first,
               current_proposal_short_track_seen_within_two_questions=second_or_first,
               pure_frame_uniform_without_replacement_short_track_seen=float(
                   hypergeom.sf(0, 1010, 10, 2)),
               object_round_robin_short_track_seen=1.0,
               not_current_neural_policy_or_semantic_accuracy=True)

    grouped = defaultdict(list)
    videos = []
    physical = set()
    inputs = []
    for case in plan['cases']:
        op = Path(case['observations'])
        qp = Path(case['temporal_queries'])
        assert sha(op) == case['observations_sha256']
        assert sha(qp) == case['temporal_queries_sha256']
        obs = np.load(op, mmap_mode='r', allow_pickle=False)
        assert obs.ndim == 2 and obs.shape[1] == 2
        ids, lengths = np.unique(obs[:, 1], return_counts=True)
        n, tracks = len(obs), len(ids)
        assert n and tracks and (lengths > 0).all()
        document = json.loads(qp.read_text())
        assert document['observations'] == n
        budgets = Counter(int(q['logical_budget']) for q in document['queries'])
        assert sum(b * count for b, count in budgets.items()) == case['logical_budget']
        by_budget = []
        for budget, count in sorted(budgets.items()):
            assert 0 < budget <= n
            # Counterfactual reference: NOT the learned, count-penalized policy.
            expected_unique = float(hypergeom.sf(0, n, lengths, budget).sum())
            round_robin_unique = min(budget, tracks)
            assert -.00000001 <= expected_unique <= round_robin_unique + .00000001
            row = dict(questions=budget, queries=count,
                       pure_frame_uniform_expected_unique_objects=expected_unique,
                       object_round_robin_unique_objects=round_robin_unique,
                       reference_unique_coverage_ratio=expected_unique / round_robin_unique)
            by_budget.append(row)
            grouped[case['pair']].extend([dict(
                budget=budget, tracks=tracks,
                reference_ratio=row['reference_unique_coverage_ratio'],
                reference_deficit=round_robin_unique - expected_unique,
                floor_minimum_track_mass=.2 * int(lengths.min()) / n,
                object_balanced_floor_minimum_track_mass=.2 / tracks,
                floor_minimum_mass_ratio=tracks * int(lengths.min()) / n,
                initial_frame_vs_object_uniform_TV=.5 * float(
                    np.abs(lengths / n - 1 / tracks).sum()))] * count)
        videos.append(dict(pair=case['pair'], target=case['target'],
                           video=case['evaluated_video'], observations=n,
                           objects=tracks, queries=sum(budgets.values()),
                           track_length_quantiles=quantiles(lengths),
                           by_budget=by_budget))
        physical.add((case['target'], case['evaluated_video']))
        inputs.append(dict(pair=case['pair'], video=case['evaluated_video'],
                           observations=str(op), observations_sha256=case['observations_sha256'],
                           queries=str(qp), queries_sha256=case['temporal_queries_sha256']))

    summary = {}
    for pair, rows in sorted(grouped.items()):
        summary[pair] = dict(
            query_instances=len(rows),
            queries_budget_less_than_object_count=sum(r['budget'] < r['tracks'] for r in rows),
            mean_pure_frame_uniform_to_object_roundrobin_unique_coverage_ratio=float(
                np.mean([r['reference_ratio'] for r in rows])),
            mean_reference_unique_object_deficit=float(np.mean([r['reference_deficit'] for r in rows])),
            reference_unique_coverage_ratio_quantiles=quantiles([r['reference_ratio'] for r in rows]),
            floor_minimum_track_mass_quantiles=quantiles([r['floor_minimum_track_mass'] for r in rows]),
            frame_to_object_balanced_floor_minimum_mass_ratio_quantiles=quantiles(
                [r['floor_minimum_mass_ratio'] for r in rows]),
            initial_frame_vs_object_uniform_TV_quantiles=quantiles(
                [r['initial_frame_vs_object_uniform_TV'] for r in rows]))
    assert len(videos) == 148 and len(physical) == 74
    assert sum(r['query_instances'] for r in summary.values()) == 3506
    assert sha(PLAN) == PLAN_SHA and sha(RUNTIME) == source_sha
    result = dict(
        status='COMPLETE_ALL148_UNLABELED_METADATA_EXPOSURE_AUDIT_NO_ALGORITHM_CHANGE',
        cases=148, physical_videos=74, query_instances=3506,
        scope='All registered candidate-frame metadata and budgets; no current learned priorities, selected questions, teacher answers, predictions, checkpoints or GT.',
        equal_priority_two_track_counterexample=toy,
        by_setting=summary, per_case=videos,
        interpretation=[
            'The .2 exploration component has per-track mass .2*remaining_track_frames/remaining_frames, not .2/remaining_tracks.',
            'The active priority and track-count penalty can change total probabilities. The pure-frame-uniform reference is not a prediction of actual active coverage.',
            'The registered UNIFORM baseline visits distinct objects before revisiting them, then selects a remaining frame within each object.',
            'Observing more frames of the same object may be useful for changing query states. Fewer distinct objects alone does not establish worse semantic or HOTA performance.',
            'No target identity or semantic labels are available here. No correct-target coverage, causal failure, new method improvement, or six-setting accuracy conclusion follows.',
            'A future learned object-then-frame acquisition must be separately registered and compared with the existing strong object-balanced uniform control, using full postseal evidence.'],
        inputs=inputs, source_sha256=sha(Path(__file__)),
        original_registration_sha256=PLAN_SHA,
        registered_proposal_source_sha256=source_sha,
        new_GPU_teacher_or_fit_calls=0, current_study_GT_reads=0,
        current_study_numerical_changes=0, actual_active_coverage_measured=False,
        goal_achieved=False)
    destination = SUP / 'fullscope_object_exposure_metadata_audit_20261126.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(destination), cases=148,
                          query_instances=3506, toy=toy,
                          by_setting={k: {a: b for a, b in v.items()
                                          if a in ['query_instances', 'queries_budget_less_than_object_count',
                                                   'mean_pure_frame_uniform_to_object_roundrobin_unique_coverage_ratio',
                                                   'mean_reference_unique_object_deficit']}
                                      for k, v in summary.items()},
                          current_study_GT_reads=0, algorithm_changed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
