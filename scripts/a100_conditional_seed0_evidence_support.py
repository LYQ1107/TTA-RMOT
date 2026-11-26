"""Apply the registered paid-evidence diagnosis to seed0 only after sealing.

Reuse the original definitions and actual paid answers. This is evaluation of
saved decisions, with no learning, new questions or target-label adaptation.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, OUT as DIAGNOSTIC
from scripts import a100_conditional_seed0_scope as scope
from scripts import a100_conditional_evidence_support as original

OUT = scope.OUT/'evidence_support'
STATE = OUT/'supervisor/state.json'
STEM = 'conditional_seed0_evidence_support'
PLAN = scope.PUBLIC/(STEM+'_plan.json')
SUCCESS = 'PASS_SINGLE_SEED_PAID_EVIDENCE_DIAGNOSIS_GOAL_ACTIVE'


def check_plan():
    plan = json.loads(PLAN.read_text())
    scope.check_registration()
    original.check_plan()
    assert plan['seeds'] == [0] and plan['condition_count'] == 6
    assert plan['methods'] == list(original.parent.METHODS)
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    return plan


def verify_seals():
    assert json.loads(scope.STATE.read_text())['status'] == scope.SUCCESS
    assert json.loads((scope.OUT/'analysis.json').read_text())['neural_seeds'] == [0]
    marker = json.loads((scope.OUT/'all_predictions_sealed_before_GT.json').read_text())
    assert marker['methods'] == 876 and marker['own_methods'] == 588 and marker['GT_open_attempts'] == 0
    for record in scope.conditions():
        for folder in (Path(record['folder']), Path(record['folder'])/'countmatched'):
            assert json.loads((folder/'completion.json').read_text())['GT_open_attempts'] == 0


def diagnose():
    verify_seals()
    started = time.perf_counter()
    methods = list(original.parent.METHODS)
    data, support, teacher = {}, {}, {}
    completed = 0
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        matrices = np.zeros((len(methods), 3, 4), dtype='int64')
        transfers = np.zeros((len(methods), 4), dtype='int64')
        errors = np.zeros((len(methods), 4), dtype='int64')
        teacher_counts = np.zeros(4, dtype='int64')
        rows_by_stratum = np.zeros(3, dtype='int64')
        positive_rows_by_stratum = np.zeros(3, dtype='int64')
        covered = uncovered = query_count = paid_count = exact_half = 0
        for video in sorted(md['videos']):
            category, anchors, raw, queries, _ = original.paid_inputs(pair, video, md)
            truth = np.load(DIAGNOSTIC/'offline_GT'/original.CANONICAL[pair]/video/'labels.npy') >= 0
            assert truth.shape == category.shape
            positive_query = truth.any(1)
            positive_asked = (truth & (category != 2)).any(1)
            query_count += len(queries)
            covered += int((positive_query & positive_asked).sum())
            uncovered += int((positive_query & ~positive_asked).sum())
            index = (anchors[:, 0], anchors[:, 1])
            anchor_truth = truth[index]
            teacher_decision = raw > .5
            teacher_correct = teacher_decision == anchor_truth
            teacher_counts += [original.confusion(anchor_truth, teacher_decision)[k] for k in ('TP', 'FP', 'FN', 'TN')]
            paid_count += len(anchors)
            exact_half += int((raw == .5).sum())
            selected = [category == i for i in range(3)]
            for i, mask in enumerate(selected):
                rows_by_stratum[i] += int(mask.sum())
                positive_rows_by_stratum[i] += int(truth[mask].sum())
            for mi, name in enumerate(methods):
                decision = np.load(original.mask_path(pair, 0, name, video), mmap_mode='r')
                assert decision.shape == truth.shape and decision.dtype == np.dtype(bool)
                for i, mask in enumerate(selected):
                    row = original.confusion(truth[mask], decision[mask])
                    matrices[mi, i] += [row[k] for k in ('TP', 'FP', 'FN', 'TN')]
                correct = decision[index] == anchor_truth
                transfers[mi] += [int((teacher_correct & correct).sum()), int((teacher_correct & ~correct).sum()),
                    int((~teacher_correct & correct).sum()), int((~teacher_correct & ~correct).sum())]
                zero = ~decision.any(1)
                missing = positive_query & zero
                errors[mi] += [int(zero.sum()), int(missing.sum()),
                    int((missing & ~positive_asked).sum()), int((missing & positive_asked).sum())]
            completed += 1
            atomic_json(STATE, dict(status='DIAGNOSING_SEED0_SEALED_PAID_SUPPORT', pid=os.getpid(),
                completed=completed, total=148, pair=pair, video=video, goal_status='ACTIVE'))
        assert np.all(matrices.sum(2) == rows_by_stratum[None, :])
        assert np.all(matrices[:, :, 0]+matrices[:, :, 2] == positive_rows_by_stratum[None, :])
        assert np.all(transfers.sum(1) == paid_count)
        assert np.all(errors[:, 1] == errors[:, 2]+errors[:, 3])
        data[pair+'_seed0'] = dict(confusion=matrices.tolist(),
            teacher_to_method=transfers.tolist(), query_errors=errors.tolist())
        support[pair] = dict(rows_by_stratum=rows_by_stratum.tolist(),
            positive_rows_by_stratum=positive_rows_by_stratum.tolist(), query_episodes=query_count,
            recoverable_positive_queries_DESIGN_positive_bag_asked=covered,
            recoverable_positive_queries_no_DESIGN_positive_bag_asked=uncovered,
            paid_mixture_nominal_anchors=paid_count, raw_mixture_exact_half=exact_half)
        teacher[pair] = teacher_counts.tolist()
    assert completed == 148 and len(data) == 6
    report = dict(status=SUCCESS, neural_seeds=[0], methods=methods,
        categories=list(original.CATEGORIES), confusion_columns=['TP', 'FP', 'FN', 'TN'],
        teacher_transition_columns=list(original.TRANSITIONS), query_error_columns=list(original.QUERY_ERRORS),
        conditions=data, DESIGN_support=support, teacher_nominal_anchor_confusion=teacher, video_groups=completed,
        canonical_candidate_membership_not_pure_teacher_semantic_accuracy=True,
        teacher_decision_rule='float32 raw same-anchor view mixture >0.5; existing hard-training >= rule unchanged',
        common_DESIGN_reference_not_overwrite_attribution_for_alternative_policies=True,
        CLIP_nominal_anchor_shared_approximation=True,
        unasked_bag_not_unseen_person_or_independent_test=True,
        raw_operating_points_not_pure_ranking_or_causal_count_decomposition=True,
        recoverable_positive_not_all_original_GT_presence=True,
        correlated_row_counts_no_significance_or_CI=True,
        references_fixed_not_all_paired_deterministic_execution=True,
        all_data_development=True, independent_blind_validation=False,
        no_method_checkpoint_threshold_or_parameter_selection=True,
        no_new_masks_fits_teacher_or_backbone_calls=True,
        after_complete_single_seed_official_and_arithmetic_reproduction=True,
        CPU_wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    (scope.PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(report, separators=(',', ':'))+'\n')
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=148, total=148, goal_status='ACTIVE'))


def main():
    check_plan()
    while True:
        current = json.loads(scope.STATE.read_text())
        if current['status'].startswith('FAILED_'):
            raise RuntimeError('Resolve single-seed parent failure before the diagnostic')
        if current['status'] == scope.SUCCESS:
            break
        atomic_json(STATE, dict(status='WAITING_COMPLETE_SINGLE_SEED_OFFICIAL_ANALYSIS_NO_GT',
            pid=os.getpid(), completed=0, goal_status='ACTIVE'))
        time.sleep(15)
    diagnose()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_SEED0_EVIDENCE_DIAGNOSIS_REQUIRES_ROOT_CAUSE',
            pid=os.getpid(), error=repr(error), goal_status='ACTIVE'))
        raise
