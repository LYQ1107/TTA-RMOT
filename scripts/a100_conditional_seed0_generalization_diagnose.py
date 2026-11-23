"""Audit train/deploy inputs and canonical temporal stability after sealing.

Two separate interpreters keep the no-GT input audit apart from the labelled
temporal diagnosis. No fitting, new predictions, teacher or backbone calls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard, OUT as GT_ROOT
from scripts import a100_conditional_seed0_scope as scope
from scripts import a100_conditional_seed0_evidence_support as evidence
from scripts.a100_conditional_standard_kd import use_verified_cache
from flex_uda.active.observation_learning_inputs import ObservationVideo

OUT = scope.OUT/'generalization_diagnosis'
METHODS = ('FULL_FINAL', 'DIRECT_PROFILE_FINAL', 'CONDITIONAL_BALANCED_FINAL',
    'LOCAL_TWO_FINAL', 'COARSE_SOFT_MEAN', 'FLAT_SOFT_FINAL')


def ready():
    assert json.loads(scope.STATE.read_text())['status'] == scope.SUCCESS
    assert json.loads(evidence.STATE.read_text())['status'] == evidence.SUCCESS
    scope.check_registration()


def inputs():
    ready()
    guard = generation_guard()
    use_verified_cache()
    summaries = {}
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        count = Counter()
        maximum = 0.
        for video, bank in sorted(md['videos'].items()):
            value = ObservationVideo(pair, video)
            _, anchors, _, _, _ = evidence.original.paid_inputs(pair, video, md)
            qi, rows = anchors.T
            ti = value.columns[rows]
            matches = (value.sample_rows[ti] == rows[:, None]) & value.present[ti]
            assert np.all(matches.sum(1) == 1)
            si = matches.argmax(1)
            expected = value.raw_bags[qi, ti, si]
            actual = value.dense_rows(qi, rows)
            assert expected.shape == actual.shape and np.isfinite(actual).all()
            difference = np.abs(expected-actual)
            maximum = max(maximum, float(difference.max()))
            count['paid_anchor_rows'] += len(rows)
            count['feature_elements'] += difference.size
            count['different_elements'] += int(np.count_nonzero(difference))
            count['different_rows'] += int((difference != 0).any(1).sum())
            total = len(value.queries)*len(value.obs)
            inside = len(value.queries)*int(value.present.sum())
            count['all_query_observation_rows'] += total
            count['current_row_is_context_anchor'] += inside
            count['current_row_not_context_anchor'] += total-inside
            count['videos'] += 1
        assert count['different_elements'] == 0, (pair, count, maximum)
        summaries[pair] = dict(count, maximum_absolute_feature_difference=maximum)
    assert not guard and sum(v['paid_anchor_rows'] for v in summaries.values()) == 55019
    report = dict(status='PASS_COMPLETE_PAID_ANCHOR_TRAIN_DEPLOY_FEATURE_ALIGNMENT',
        settings=summaries, dimensions=1089, exact_numeric_equal=True,
        feature_version='Same original historicalPCD/denseRGB used by all registered learners',
        context_relation='Every teacher training current observation is one of its three bag anchors; most dense inference rows need not be anchors',
        no_claim_context_membership_shift_caused_quality_failure=True,
        no_GT_HOTA_or_method_outputs_read=True, GT_open_attempts=0,
        new_fits_predictions_teacher_backbone_calls=0, goal_status='ACTIVE')
    atomic_json(OUT/'input_alignment.json', report)
    (scope.PUBLIC/'conditional_seed0_input_alignment.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(report), flush=True)


def temporal():
    ready()
    summaries = {}
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        counts = Counter()
        methods = {name:Counter() for name in METHODS}
        for video, bank in sorted(md['videos'].items()):
            obs = np.asarray(bank['observations'], dtype=int)
            labels = np.load(GT_ROOT/'offline_GT'/evidence.original.CANONICAL[pair]/video/'labels.npy')
            masks = {m:np.load(evidence.original.mask_path(pair, 0, m, video), mmap_mode='r') for m in METHODS}
            assert all(v.shape == labels.shape for v in masks.values())
            for track in np.unique(obs[:, 1]):
                ids = np.flatnonzero(obs[:, 1] == track)
                left, right = ids[:-1], ids[1:]
                consecutive = obs[right, 0]-obs[left, 0] == 1
                left, right = left[consecutive], right[consecutive]
                old, new = labels[:, left], labels[:, right]
                positive = (old >= 0) & (old == new)
                negative = (old < 0) & (new < 0)
                change = (old >= 0) != (new >= 0)
                counts.update(stable_same_positive_ID_pairs=int(positive.sum()),
                    stable_negative_membership_pairs=int(negative.sum()),
                    canonical_membership_change_pairs=int(change.sum()))
                for name, mask in masks.items():
                    a, b = mask[:, left], mask[:, right]
                    flip = a != b
                    methods[name].update(
                        positive_ID_stable_output_flips=int((positive & flip).sum()),
                        negative_membership_stable_output_flips=int((negative & flip).sum()),
                        stable_positive_missing_endpoints=int((positive & ~a).sum()+(positive & ~b).sum()),
                        stable_negative_positive_endpoints=int((negative & a).sum()+(negative & b).sum()),
                        membership_change_output_flips=int((change & flip).sum()),
                        membership_change_both_endpoints_correct=int((change & (a == (old >= 0)) & (b == (new >= 0))).sum()))
            counts['videos'] += 1
        summaries[pair] = dict(denominators=dict(counts), methods={m:dict(v) for m, v in methods.items()})
    report = dict(status='PASS_POSTSEAL_NEW_COHORT_TEMPORAL_MEMBERSHIP_DIAGNOSIS',
        seeds=[0], settings=summaries, methods=list(METHODS),
        consecutive_frames_same_provided_TrackID_only=True,
        positive_stability_requires_same_canonical_positive_GT_ID=True,
        negative_membership_does_not_certify_same_physical_identity=True,
        canonical_matching_changes_not_pure_semantic_state_changes=True,
        adjacent_pairs_and_endpoints_correlated_no_significance_CI=True,
        coarse_heads_constant_per_given_TrackID_zero_flips_by_construction=True,
        no_new_predictions_fits_teacher_or_backbone_calls=True,
        all_outputs_and_official_analysis_complete_before_this_GT=True,
        all_data_development=True, independent_blind_validation=False, goal_status='ACTIVE')
    atomic_json(OUT/'temporal.json', report)
    (scope.PUBLIC/'conditional_seed0_temporal_diagnosis.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('inputs', 'temporal'), required=True)
    args = parser.parse_args()
    inputs() if args.mode == 'inputs' else temporal()
