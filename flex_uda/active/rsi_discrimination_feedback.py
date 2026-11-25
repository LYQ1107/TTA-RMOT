"""Conditional teacher discrimination as curriculum feedback.

For each query, the best constant distribution on a paid probe is its mean
teacher opinion. Its cross entropy is H(mean Q). Gibbs' inequality bounds
positive learner skill by H(mean Q)-mean H(Q). Thus merely fitting a No rate
cannot earn positive skill, and a frozen learner cannot earn progress.

These are statements about the teacher scoring rule, not RMOT correctness.
"""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np


def paired_probe_reservation(actions, quotas, fraction=.1, rounds=3):
    """Reserve different tracks for the same query/view, before any answers."""
    groups = defaultdict(lambda: defaultdict(list))
    for uid, action in actions.items():
        groups[int(action['local_query'])][int(action['local_track'])].append(uid)
    def key(uid):
        return hashlib.sha256(('rsi-conditional-probe-v1:'+str(uid)).encode()).digest()
    qorder = sorted(groups, key=lambda q: min(key(u) for us in groups[q].values() for u in us))
    maximum = (sum(quotas.values())-len(quotas))//2
    wanted = min(maximum, max(rounds, round(sum(quotas.values())*fraction/2)))
    remaining = {q: sum(map(len, g.values())) for q, g in groups.items()}
    reserved, counts, pairs = set(), Counter(), []
    while len(pairs) < wanted:
        progress = False
        for qi in qorder:
            if len(pairs) == wanted:
                break
            if counts[qi]+2 >= quotas[qi]:
                continue
            tracks = sorted((t for t in groups[qi] if (qi, t) not in reserved),
                            key=lambda t: min(key(u) for u in groups[qi][t]))
            chosen = None
            for i, first in enumerate(tracks):
                a = groups[qi][first]
                av = {actions[u]['case']['view'] for u in a}
                for second in tracks[i+1:]:
                    b = groups[qi][second]
                    views = av & {actions[u]['case']['view'] for u in b}
                    if not views:
                        continue
                    after = remaining[qi]-len(a)-len(b)
                    if after < quotas[qi]-counts[qi]-2:
                        continue
                    possibilities = [(min((u for u in a if actions[u]['case']['view'] == v), key=key),
                                      min((u for u in b if actions[u]['case']['view'] == v), key=key)) for v in views]
                    chosen = min(possibilities, key=lambda p: key(p[0])+key(p[1]))
                    break
                if chosen:
                    break
            if chosen:
                a, b = chosen
                ta, tb = int(actions[a]['local_track']), int(actions[b]['local_track'])
                assert ta != tb and actions[a]['case']['view'] == actions[b]['case']['view']
                reserved.update(((qi, ta), (qi, tb)))
                remaining[qi] -= len(groups[qi][ta])+len(groups[qi][tb])
                counts[qi] += 2
                pairs.append((a, b))
                progress = True
        if not progress:
            break
    pair_rounds = [pairs[i::rounds] for i in range(rounds)]
    probes = [[u for pair in pairs for u in pair] for pairs in pair_rounds]
    schedule = [{q: (quotas[q]-counts[q])//rounds+
                 int(stage < (quotas[q]-counts[q]) % rounds) for q in quotas} for stage in range(rounds)]
    for q, budget in quotas.items():
        assert sum(s[q] for s in schedule)+counts[q] == budget
        assert schedule[0][q] >= 1
        assert remaining[q] >= budget-counts[q]
    return dict(probes=probes, pairs=pair_rounds, groups=reserved,
                training_schedule=schedule, requested_probe_count=2*wanted,
                actual_probe_count=2*len(pairs),
                fresh_probe_rounds=sum(bool(p) for p in probes),
                same_query_same_view_different_tracks=True)


def teacher_discrimination_summary(probe):
    """A strong diagnostic null, fitted to the paid probe itself, never GT.

    This oracle intercept is only a reward baseline. It is never a student
    input, a deployed RMOT predictor, or training supervision in this round.
    """
    if not probe:
        return None
    target = np.asarray([r['joint_target'] for r in probe], dtype='float64')
    assert target.ndim == 2 and target.shape[-1] == 4 and np.all(target >= 0)
    target /= target.sum(-1, keepdims=True)
    query = np.asarray([r['local_query'] for r in probe])
    null, floor, groups = [], [], []
    for qi in np.unique(query):
        rows = np.flatnonzero(query == qi)
        own = target[rows]
        members = [probe[int(i)] for i in rows]
        hown = -float((own*np.log(own.clip(min=1e-300))).sum(-1).mean())
        views = [r['case']['view'] for r in members]
        hmean = 0.
        for view in sorted(set(views)):
            chosen = np.asarray([v == view for v in views])
            assert len({r['local_track'] for r, keep in zip(members, chosen) if keep}) >= 2
            subset = own[chosen]
            mean = subset.mean(0)
            hv = -float((mean*np.log(mean.clip(min=1e-300))).sum())
            hf = -float((subset*np.log(subset.clip(min=1e-300))).sum(-1).mean())
            assert hv >= hf-1e-12
            hmean += float(chosen.mean())*hv
            groups.append(dict(local_query=int(qi), view=view, observations=int(chosen.sum()),
                               constant_entropy=hv, teacher_entropy_floor=hf,
                               conditional_information=max(hv-hf, 0.)))
        null.append(hmean)
        floor.append(hown)
    return dict(best_constant_CE=float(np.mean(null)), teacher_entropy_floor=float(np.mean(floor)),
                conditional_information=max(float(np.mean(null)-np.mean(floor)), 0.),
                macro_query_weighting=True, null_conditioning='query and requested view', groups=groups,
                null_fitted_to_paid_teacher_probe_not_target_GT=True)


def discrimination_progress(before, after, summary):
    if summary is None:
        return dict(reward=0., conditional_skill_before=None, conditional_skill_after=None)
    assert math.isfinite(before) and math.isfinite(after)
    null = summary['best_constant_CE']
    maximum = summary['conditional_information']
    a, b = max(null-before, 0.), max(null-after, 0.)
    assert a <= maximum+1e-9 and b <= maximum+1e-9
    # Numerical noise at an entropy floor is not curriculum progress.
    if maximum < 1e-12:
        a = b = 0.
    return dict(reward=math.tanh(b-a), conditional_skill_before=a,
                conditional_skill_after=b, teacher_conditional_information=maximum,
                best_constant_CE=null, teacher_entropy_floor=summary['teacher_entropy_floor'],
                reward_basis='GAIN_IN_CONDITIONAL_TEACHER_SKILL',
                pure_query_constant_predictor_positive_skill_is_impossible=True,
                not_a_claim_of_true_RMOT_semantic_gain=True)
