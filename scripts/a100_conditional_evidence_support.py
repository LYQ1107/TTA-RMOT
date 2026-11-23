"""Diagnose sealed RMOT outputs relative to actual paid teacher evidence.

Run only after the complete deterministic study and its official analysis.
Candidate membership is the existing canonical offline diagnostic, not
teacher semantic accuracy or a replacement for official HOTA/CountMatched.
No fitting, teacher/backbone inference, threshold selection or new masks.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, OUT as DIAGNOSTIC, generation_guard
from scripts import a100_conditional_deterministic_repair as parent
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA
from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL, confusion
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS

OUT = SIX/'conditional_evidence_support_20261123'
STATE = OUT/'supervisor/state.json'
PLAN = PUBLIC/'conditional_evidence_support_plan.json'
SUCCESS = 'PASS_COMPLETE_CONDITIONAL_EVIDENCE_SUPPORT_GOAL_ACTIVE'
CATEGORIES = ('PAID_NOMINAL_ANCHOR', 'OTHER_ROW_IN_DESIGN_ASKED_BAG', 'DESIGN_UNASKED_BAG')
TRANSITIONS = ('BOTH_CORRECT', 'TEACHER_CORRECT_METHOD_WRONG', 'TEACHER_WRONG_METHOD_CORRECT', 'BOTH_WRONG')
QUERY_ERRORS = ('ZERO_OUTPUT', 'RECOVERABLE_POSITIVE_ZERO_OUTPUT',
    'RECOVERABLE_POSITIVE_ZERO_OUTPUT_NO_DESIGN_POSITIVE_BAG_ASKED',
    'RECOVERABLE_POSITIVE_ZERO_OUTPUT_DESIGN_POSITIVE_BAG_ASKED')


def check_plan():
    plan = json.loads(PLAN.read_text())
    parent.check_registration()
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    assert plan['methods'] == list(parent.METHODS)
    assert plan['categories'] == list(CATEGORIES)
    assert plan['wait_for_complete_parent_official_analysis']
    return plan


def paid_inputs(pair, video, md):
    bank = md['videos'][video]
    obs = np.asarray(bank['observations'], dtype=int)
    queries = [q for q in md['episodes'] if q['video_id'] == video]
    tracks = np.unique(obs[:, 1])
    positions = {tuple(map(int, r)):i for i, r in enumerate(obs)}
    doc = json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
    assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
    seen, grouped, asked = {}, defaultdict(list), defaultdict(set)
    for row in doc['answers']:
        rid, probability = row['request_id'], float(row['yes_probability'])
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        if rid in seen:
            assert seen[rid] == probability
            continue
        seen[rid] = probability
        qi, track, frame = int(row['local_query']), int(row['track_id']), int(row['frame_id'])
        assert queries[qi]['expression_id'] == row['query_id']
        assert int(tracks[int(row['local_track'])]) == track
        grouped[qi, positions[frame, track]].append(probability)
        asked[qi].add(track)
    category = np.full((len(queries), len(obs)), 2, dtype='uint8')
    for qi, tracks in asked.items():
        category[qi, np.isin(obs[:, 1], list(tracks))] = 1
    anchors = np.asarray(sorted(grouped), dtype=int).reshape(-1, 2)
    raw = np.asarray([np.mean(grouped[tuple(k)]) for k in anchors], dtype='float32')
    if len(anchors):
        category[anchors[:, 0], anchors[:, 1]] = 0
    return category, anchors, raw, queries, obs


def mask_path(pair, seed, name, video):
    root = parent.reference_root(name, pair, seed) if name in parent.REFERENCES else parent.OUT/pair/f'seed{seed}'
    return root/name/'videos'/video/'mask.npy'


def validate_inputs():
    """Header/QA checks only; no target truth or decision values are read."""
    check_plan()
    guard = generation_guard()
    cases = {}
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        video = sorted(md['videos'])[0]
        category, anchors, raw, _, _ = paid_inputs(pair, video, md)
        checked, pending = [], []
        receipt = parent.OUT/pair/'seed0/videos'/video/'receipt.json'
        for name in parent.METHODS:
            if name in parent.OWN and not receipt.is_file():
                pending.append(name)
                continue
            value = np.load(mask_path(pair, 0, name, video), mmap_mode='r')
            assert value.shape == category.shape and value.dtype == np.dtype(bool)
            checked.append(name)
        cases[pair] = dict(video=video, shape=list(category.shape), paid_nominal_anchors=len(anchors),
            header_checked_methods=checked, own_methods_pending_complete_first_video=pending,
            exact_half_mixture_targets=int((raw == .5).sum()))
    assert not guard
    report = dict(status='PASS_EVIDENCE_SUPPORT_INPUT_HEADERS_NO_GT', cases=cases,
        no_target_GT_or_prediction_decision_values_read=True,
        new_fits_teacher_backbone_calls=0, GT_open_attempts=0,
        not_full_study_acceptance=True, goal_status='ACTIVE')
    atomic_json(PUBLIC/'conditional_evidence_support_input_audit.json', report)
    print(json.dumps(report), flush=True)


def verify_complete_seals():
    assert json.loads(parent.STATE.read_text())['status'] == parent.SUCCESS
    assert (parent.OUT/'analysis.json').is_file()
    marker = json.loads((parent.OUT/'all_predictions_sealed_before_GT.json').read_text())
    assert marker['methods'] == 1764 and marker['GT_open_attempts'] == 0
    for root, expected in parent.PARENTS:
        proof = json.loads((root/'all_predictions_sealed_before_GT.json').read_text())
        assert proof['methods'] == expected and proof['GT_open_attempts'] == 0
    for record in parent.conditions():
        root = Path(record['folder'])
        for folder in (root, root/'countmatched'):
            assert json.loads((folder/'completion.json').read_text())['GT_open_attempts'] == 0


def diagnose():
    check_plan()
    verify_complete_seals()
    started = time.perf_counter()
    methods = list(parent.METHODS)
    data, support, teacher = {}, {}, {}
    completed = 0
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        matrices = {s:np.zeros((len(methods), len(CATEGORIES), 4), dtype='int64') for s in SEEDS}
        transfers = {s:np.zeros((len(methods), 4), dtype='int64') for s in SEEDS}
        errors = {s:np.zeros((len(methods), len(QUERY_ERRORS)), dtype='int64') for s in SEEDS}
        teacher_counts = np.zeros(4, dtype='int64')
        rows_by_stratum = np.zeros(3, dtype='int64')
        positive_rows_by_stratum = np.zeros(3, dtype='int64')
        covered = uncovered = query_count = paid_count = exact_half = 0
        for video in sorted(md['videos']):
            category, anchors, raw, queries, obs = paid_inputs(pair, video, md)
            truth = np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy') >= 0
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
            teacher_counts += [confusion(anchor_truth, teacher_decision)[k] for k in ('TP', 'FP', 'FN', 'TN')]
            paid_count += len(anchors)
            exact_half += int((raw == .5).sum())
            selected = [category == i for i in range(3)]
            for i, mask in enumerate(selected):
                rows_by_stratum[i] += int(mask.sum())
                positive_rows_by_stratum[i] += int(truth[mask].sum())
            for seed in SEEDS:
                for mi, name in enumerate(methods):
                    decision = np.load(mask_path(pair, seed, name, video), mmap_mode='r')
                    assert decision.shape == truth.shape and decision.dtype == np.dtype(bool)
                    for i, mask in enumerate(selected):
                        row = confusion(truth[mask], decision[mask])
                        matrices[seed][mi, i] += [row[k] for k in ('TP', 'FP', 'FN', 'TN')]
                    correct = decision[index] == anchor_truth
                    transfers[seed][mi] += [int((teacher_correct & correct).sum()), int((teacher_correct & ~correct).sum()),
                        int((~teacher_correct & correct).sum()), int((~teacher_correct & ~correct).sum())]
                    zero = ~decision.any(1)
                    missing = positive_query & zero
                    errors[seed][mi] += [int(zero.sum()), int(missing.sum()),
                        int((missing & ~positive_asked).sum()), int((missing & positive_asked).sum())]
            completed += 1
            atomic_json(STATE, dict(status='DIAGNOSING_COMPLETE_SEALED_RAW_OUTPUTS', pid=os.getpid(),
                completed=completed, total=148, pair=pair, video=video, goal_status='ACTIVE'))
        for seed in SEEDS:
            assert np.all(matrices[seed].sum(2) == rows_by_stratum[None, :])
            assert np.all(matrices[seed][:, :, 0]+matrices[seed][:, :, 2] == positive_rows_by_stratum[None, :])
            assert np.all(transfers[seed].sum(1) == paid_count)
            assert np.all(errors[seed][:, 1] == errors[seed][:, 2]+errors[seed][:, 3])
            data[pair+f'_seed{seed}'] = dict(confusion=matrices[seed].tolist(),
                teacher_to_method=transfers[seed].tolist(), query_errors=errors[seed].tolist())
        support[pair] = dict(rows_by_stratum=rows_by_stratum.tolist(), positive_rows_by_stratum=positive_rows_by_stratum.tolist(),
            query_episodes=query_count, recoverable_positive_queries_DESIGN_positive_bag_asked=covered,
            recoverable_positive_queries_no_DESIGN_positive_bag_asked=uncovered,
            paid_mixture_nominal_anchors=paid_count, raw_mixture_exact_half=exact_half)
        teacher[pair] = teacher_counts.tolist()
    assert completed == 148 and len(data) == 18
    report = dict(status=SUCCESS, methods=methods, categories=list(CATEGORIES),
        confusion_columns=['TP', 'FP', 'FN', 'TN'], teacher_transition_columns=list(TRANSITIONS),
        query_error_columns=list(QUERY_ERRORS), conditions=data, DESIGN_support=support,
        teacher_nominal_anchor_confusion=teacher, video_groups=completed,
        canonical_candidate_membership_not_pure_teacher_semantic_accuracy=True,
        teacher_decision_rule='float32 raw same-anchor view mixture >0.5; exact halves reported; existing hard-training >= rule unchanged',
        teacher_transition_reference='Common DESIGN mixture; alternative-policy arms can have different own answers, so their transitions are not overwrite attribution',
        teacher_answers_shared_across_seeds_not_independent_recollections=True,
        CLIP_nominal_anchor_shared_approximation=True,
        common_DESIGN_strata_for_all_policies_not_randomized_groups=True,
        unasked_means_current_query_bag_has_no_own_DESIGN_answer_not_unseen_person_or_independent_test=True,
        raw_operating_points_not_pure_ranking_or_causal_count_decomposition=True,
        official_dual_CountMatched_and_HOTA_remain_separate_parent_analysis=True,
        recoverable_positive_not_all_original_GT_presence=True,
        row_counts_correlated_no_CI_or_independent_sample_claim=True,
        references_fixed_not_all_paired_deterministic_execution=True,
        all_data_development=True, independent_blind_validation=False,
        no_method_checkpoint_threshold_or_parameter_selection=True,
        no_new_masks_fits_teacher_or_backbone_calls=True, after_complete_parent_official_analysis=True,
        CPU_wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    (PUBLIC/'conditional_evidence_support_analysis.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=148, total=148, goal_status='ACTIVE'))


def main():
    check_plan()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='WAITING_COMPLETE_PARENT_OFFICIAL_ANALYSIS_NO_GT', pid=os.getpid(),
        completed=0, total=148, goal_status='ACTIVE'))
    while True:
        state = json.loads(parent.STATE.read_text())
        if state['status'].startswith('FAILED_'):
            raise RuntimeError('Resolve parent deterministic study before diagnosis')
        if state['status'] == parent.SUCCESS:
            break
        time.sleep(30)
    diagnose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args()
    if args.validate_inputs:
        validate_inputs()
    else:
        try:
            main()
        except Exception as error:
            old = json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_POSTSEAL_EVIDENCE_SUPPORT_DIAGNOSIS', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, old)
            raise
