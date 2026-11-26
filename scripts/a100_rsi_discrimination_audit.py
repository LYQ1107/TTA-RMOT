"""Check conditional-skill feedback and paired probes without real labels."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.rsi_curriculum import question_inventory
from flex_uda.active.rsi_discrimination_feedback import (
    paired_probe_reservation, teacher_discrimination_summary, discrimination_progress)
from scripts.a100_joint_observation_collect import OUT as COLLECTION
from scripts.a100_six_direction import PUBLIC, OUT as SIX


def row(track, target):
    return dict(local_query=0, local_track=track, case=dict(view='CROP'), joint_target=target)


def main():
    guard = generation_guard()
    negative = teacher_discrimination_summary([
        row(1, [0, 0, 0, 1]), row(2, [0, 0, 0, 1])])
    contrasting = teacher_discrimination_summary([
        row(1, [0, 0, 0, 1]), row(2, [0, 0, 1, 0])])
    contract = {
        'LEARN_NO_ONLY': discrimination_progress(math.log(4), .001, negative),
        'LEARN_OPTIMAL_CONSTANT_ONLY': discrimination_progress(math.log(4), math.log(2), contrasting),
        'FROZEN_ALREADY_DISCRIMINATIVE': discrimination_progress(.1, .1, contrasting),
        'LEARN_ACTUAL_TEACHER_DISCRIMINATION': discrimination_progress(math.log(4), .01, contrasting)}
    assert all(abs(contract[k]['reward']) < 1e-12 for k in (
        'LEARN_NO_ONLY', 'LEARN_OPTIMAL_CONSTANT_ONLY', 'FROZEN_ALREADY_DISCRIMINATIVE'))
    assert contract['LEARN_ACTUAL_TEACHER_DISCRIMINATION']['reward'] > 0
    rows = {}
    for pair, (_, target) in PAIRS.items():
        own = []
        for video in sorted(metadata(target)['videos']):
            doc = json.loads((COLLECTION/'inputs'/target/video/'manifest.json').read_text())
            actions, quotas = question_inventory(doc, pair)
            reserve = paired_probe_reservation(actions, quotas)
            own.append(dict(video=video, teacher_budget=sum(quotas.values()),
                            pair_count=sum(map(len, reserve['pairs'])),
                            paid_probe_bundles=reserve['actual_probe_count'],
                            fresh_probe_rounds=reserve['fresh_probe_rounds'],
                            all_queries_first_round=min(reserve['training_schedule'][0].values()) >= 1,
                            same_query_same_view_distinct_tracks=True))
        rows[pair] = own
    assert not guard
    report = dict(status='PASS_CONDITIONAL_SKILL_MATH_AND_UNLABELLED_PAIRED_PROBE_AUDIT',
        reward_contract=contract,
        settings={p: dict(videos=len(r), teacher_budget=sum(v['teacher_budget'] for v in r),
                          paid_probe_bundles=sum(v['paid_probe_bundles'] for v in r),
                          min_fresh_probe_rounds=min(v['fresh_probe_rounds'] for v in r),
                          videos_with_all_three_paired_rounds=sum(v['fresh_probe_rounds'] == 3 for v in r),
                          all_queries_keep_first_round_training=all(v['all_queries_first_round'] for v in r))
                  for p, r in rows.items()},
        real_teacher_answers_read=0, GT_open_attempts=0, model_fits=0,
        semantic_or_HOTA_improvement_proven=False,
        classical_support='Proper log scoring: best query/view constant risk H(mean Q); conditional skill bounded by H(mean Q)-mean H(Q). Fixed probe null cannot manufacture frozen-solver progress.',
        null_is_fitted_to_paid_teacher_probe_only_for_feedback_not_deployed=True,
        future_actor_optimization='Reward magnitude must matter; use an explicitly registered SGD policy update rather than Adam amplifying tiny teacher-probability differences.',
        source_sha256={name: file_sha(ROOT/name) for name in (
            'scripts/a100_rsi_discrimination_audit.py',
            'flex_uda/active/rsi_discrimination_feedback.py')}, goal_status='ACTIVE')
    out = SIX/'rsi_discrimination_audit_20261123'
    atomic_json(out/'per_video_probe_pairs.json', rows)
    atomic_json(out/'analysis.json', report)
    atomic_json(PUBLIC/'rsi_discrimination_feedback_audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
