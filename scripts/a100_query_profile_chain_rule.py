"""Audit the exact KL decomposition underlying the current profile objective.

This is algebra on actual frozen teacher targets and hypothetical scores.
It does not read task predictions or GT, fit a head, change a registered
method, or establish RMOT quality. Classical conditional decomposition
explains what the loss does, without claiming it as a new theorem.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.conditional_observation import weighted_bernoulli_loss
from flex_uda.active.query_profile import supported_query_profile_kl
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA
from scripts.a100_six_direction import PUBLIC

VIDEOS = dict(C1=25, C2=4, C3=45, C4=45, C5=4, C6=25)
POLICIES = ('DESIGN', 'PRIOR', 'UNIFORM', 'COVERAGE_ONLY')
FAMILIES = ('TEACHER_EXACT', 'GROUP_SCALE', 'CONTRAST_PERTURBATION')
TOLERANCE = 5e-12


def teacher_arrays(path):
    """Rebuild only the scalar inputs used by the original prefix loss."""
    doc = json.loads(path.read_text())
    assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
    seen, by_anchor = {}, defaultdict(list)
    for row in doc['answers']:
        probability = float(row['yes_probability'])
        rid = row['request_id']
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        if rid in seen:
            assert seen[rid] == probability
            continue
        seen[rid] = probability
        key = int(row['local_query']), int(row['local_track']), int(row['frame_id'])
        by_anchor[key].append(probability)
    keys = np.asarray(sorted(by_anchor), dtype='int64')
    raw = np.asarray([np.mean(by_anchor[tuple(key)]) for key in keys], dtype='float32')
    _, bag = np.unique(keys[:, :2], axis=0, return_inverse=True)
    confidence = (1+3*np.abs(2*raw-1)).astype('float32')
    weight = (confidence/np.bincount(bag)[bag]).astype('float32')
    return raw, np.clip(raw, .02, .98).astype('float32'), weight, keys[:, 0]


def bernoulli_kl(p, q):
    return p*(np.log(p)-np.log(q))+(1-p)*(np.log1p(-p)-np.log1p(-q))


def decompose(raw, target, weight, group, family):
    p, w = target.astype('float64'), weight.astype('float64')
    if family == 'TEACHER_EXACT':
        q = p.copy()
    elif family == 'GROUP_SCALE':
        q = p*(.4+.1*(group % 3))
    else:
        row = np.arange(len(p))
        z = np.log(p)-np.log1p(-p)+.7*np.sin(row*.37)+.35*np.cos(group*.61)
        q = 1/(1+np.exp(-z))
    counts = np.bincount(group)
    maximum = np.full(len(counts), -np.inf)
    np.maximum.at(maximum, group, raw)
    selected = ((counts >= 2) & (maximum >= .5))[group]
    all_weight = float(w.sum())
    selected_weight = float(w[selected].sum())
    point = float(np.dot(w, bernoulli_kl(p, q))/all_weight)
    logq = torch.as_tensor(np.log(q))
    tt, ww = torch.as_tensor(p), torch.as_tensor(w)
    actual = float(supported_query_profile_kl(logq, tt,
        torch.as_tensor(raw.astype('float64')), torch.as_tensor(group), ww))
    z = np.log(q)-np.log1p(-q)
    entropy = float(np.dot(w, -p*np.log(p)-(1-p)*np.log1p(-p))/all_weight)
    point_code = float(weighted_bernoulli_loss(torch.as_tensor(z), tt, ww))-entropy
    assert abs(point_code-point) <= TOLERANCE
    if selected_weight == 0:
        assert abs(actual) <= TOLERANCE
        return dict(point_KL=point, selected_weight_fraction=0., profile_KL=actual,
            identity_error=0., code_error=0., bound_violation=0.)
    pp, qq, sw = p[selected], q[selected], w[selected]
    _, sg = np.unique(group[selected], return_inverse=True)
    W = np.bincount(sg, weights=sw)
    M = np.bincount(sg, weights=sw*pp)
    N = np.bincount(sg, weights=sw*qq)
    mass = float(np.dot(W, bernoulli_kl(M/W, N/W))/selected_weight)
    positive = float(np.sum(sw*pp*(np.log(sw*pp/M[sg])-np.log(sw*qq/N[sg])))/selected_weight)
    negative = float(np.sum(sw*(1-pp)*(np.log(sw*(1-pp)/(W-M)[sg])-
        np.log(sw*(1-qq)/(W-N)[sg])))/selected_weight)
    selected_point = float(np.dot(sw, bernoulli_kl(pp, qq))/selected_weight)
    identity_error = abs(selected_point-mass-positive-negative)
    code_error = abs(actual-positive)
    bound_violation = max(0., actual-point*all_weight/selected_weight)
    assert min(mass, positive, negative, point) >= -TOLERANCE
    assert max(identity_error, code_error, bound_violation) <= TOLERANCE
    if family == 'TEACHER_EXACT':
        assert max(abs(point), abs(actual)) <= TOLERANCE
    elif family == 'GROUP_SCALE':
        assert abs(actual) <= TOLERANCE and point > 0
    return dict(point_KL=point, selected_weight_fraction=selected_weight/all_weight,
        profile_KL=actual, positive_KL=positive, mass_KL=mass, negative_KL=negative,
        identity_error=identity_error, code_error=code_error, bound_violation=bound_violation)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    guard = generation_guard()
    summaries = {}
    maximum_identity = maximum_code = maximum_bound = 0.
    cases = prefixes = 0
    for pair, expected in VIDEOS.items():
        videos = sorted(path.parent.name for path in (ANSWERS/'policy_answers'/pair).glob('*/DESIGN_answers.json'))
        assert len(videos) == expected, (pair, len(videos))
        for policy in POLICIES:
            records, offset = [], 0
            row = dict(video_prefixes=0, numerical_cases=0,
                minimum_selected_weight_fraction=1., maximum_identity_error=0.,
                maximum_code_error=0., maximum_bound_violation=0.,
                group_scale_point_KL_range=[float('inf'), 0.],
                maximum_group_scale_profile_magnitude=0.)
            for video in videos:
                raw, p, w, g = teacher_arrays(ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json'))
                records.append((raw, p, w, g+offset))
                offset += int(g.max())+1
                values = [np.concatenate([r[i] for r in records]) for i in range(4)]
                for family in FAMILIES:
                    result = decompose(*values, family)
                    cases += 1
                    row['numerical_cases'] += 1
                    row['minimum_selected_weight_fraction'] = min(row['minimum_selected_weight_fraction'], result['selected_weight_fraction'])
                    for name in ('identity_error', 'code_error', 'bound_violation'):
                        row['maximum_'+name] = max(row['maximum_'+name], result[name])
                    if family == 'GROUP_SCALE':
                        row['group_scale_point_KL_range'][0] = min(row['group_scale_point_KL_range'][0], result['point_KL'])
                        row['group_scale_point_KL_range'][1] = max(row['group_scale_point_KL_range'][1], result['point_KL'])
                        row['maximum_group_scale_profile_magnitude'] = max(row['maximum_group_scale_profile_magnitude'], abs(result['profile_KL']))
                row['video_prefixes'] += 1
                prefixes += 1
            summaries[pair+'_'+policy] = row
            maximum_identity = max(maximum_identity, row['maximum_identity_error'])
            maximum_code = max(maximum_code, row['maximum_code_error'])
            maximum_bound = max(maximum_bound, row['maximum_bound_violation'])
    assert not guard and prefixes == 592 and cases == 1776
    report = dict(status='PASS_QUERY_PROFILE_CLASSICAL_CHAIN_RULE_AUDIT',
        scope='Algebraic loss decomposition using actual paid teacher targets/weights/groups and hypothetical score vectors; not actual model quality',
        video_prefixes=prefixes, numerical_cases=cases, score_families=list(FAMILIES),
        maximum_identity_error=maximum_identity, maximum_code_error=maximum_code,
        maximum_bound_violation=maximum_bound, numerical_tolerance=TOLERANCE,
        same_pointwise_and_profile_weights_required=True,
        equal_targets_and_disjoint_fixed_groups_required=True,
        bounds='K_eligible=mass_KL+profile_KL+negative_KL; 0<=profile_KL<=K_all/f, f=eligible_weight/all_weight>0',
        no_teacher_accuracy_or_HOTA_guarantee=True, not_a_new_theorem=True,
        no_registered_method_or_primary_changed=True, no_task_prediction_or_current_quality_read=True,
        no_new_fits_teacher_or_backbone_forwards=True, GT_open_attempts=len(guard),
        summaries=summaries, CPU_wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    path = PUBLIC/'query_profile_chain_rule_audit.json'
    atomic_json(path, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'summaries'}), flush=True)


if __name__ == '__main__':
    main()
