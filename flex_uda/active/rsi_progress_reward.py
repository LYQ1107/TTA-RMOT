"""Progress credit beyond a teacher-response-prior learner.

The null learner uses only already requested training replies, no visual
features and no probe targets. It is an acquisition reward control, not an
RMOT output calibration rule or an oracle for semantic correctness.
"""
import math
import numpy as np


def teacher_prior_cross_entropy(probe, current_record=None, past_records=()):
    if not probe:
        return None
    available = [r['joint_target'] for r in past_records]
    if current_record is not None:
        available.append(current_record['joint_target'])
    fallback = np.concatenate(available).mean(0) if available else np.full(4, .25)
    probabilities = []
    for row in probe:
        qi = int(row['local_query'])
        own = current_record['query_groups'] == qi if current_record is not None else None
        p = current_record['joint_target'][own].mean(0) if own is not None and own.any() else fallback
        # Full support for possible singleton/letter-probability zeros. This
        # constant is fixed before any new task evaluation, never tuned on GT.
        p = np.maximum(p, 1e-6)
        probabilities.append(p/p.sum())
    target = np.asarray([r['joint_target'] for r in probe], dtype='float64')
    target /= target.sum(-1, keepdims=True)
    ce = -(target*np.log(np.asarray(probabilities))).sum(-1)
    query = np.asarray([r['local_query'] for r in probe])
    return float(np.mean([ce[query == q].mean() for q in np.unique(query)]))


def skill_progress(student_before, student_after, null_before, null_after):
    if student_before is None:
        return dict(reward=0., student_gain=None, teacher_prior_gain=None,
                    supported_gain=None, reward_basis='PROGRESS_BEYOND_TEACHER_PRIOR')
    values = (student_before, student_after, null_before, null_after)
    assert all(v is not None and math.isfinite(v) for v in values)
    gain = student_before-student_after
    prior_gain = null_before-null_after
    # A worse null predictor cannot manufacture progress. A frozen solver
    # receives exactly zero; regressions remain negative. Positive credit
    # requires solver improvement beyond any positive null-model gain.
    supported = max(gain-max(prior_gain, 0.), 0.)+min(gain, 0.)
    return dict(reward=math.tanh(supported), student_gain=gain,
                teacher_prior_gain=prior_gain, supported_gain=supported,
                null_CE_before=null_before, null_CE_after=null_after,
                reward_basis='PROGRESS_BEYOND_TEACHER_PRIOR',
                not_a_HOTA_or_semantic_gain_guarantee=True)
