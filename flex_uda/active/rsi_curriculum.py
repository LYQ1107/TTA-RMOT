"""Task-level challenger/solver co-evolution with a paid teacher feedback loop.

The challenger observes images/student opinions, not unrequested teacher
answers. Its signal is one-step progress on fresh teacher probe groups.
This is an RSI-inspired learning procedure, not evidence of semantic accuracy
or a software intelligence explosion. Ground truth is never an input.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn


def question_inventory(manifest, pair):
    """Only question metadata enters the action space; no answer file is read."""
    by_policy = {}
    actions = {}
    for policy in ('DESIGN', 'UNIFORM'):
        rows = manifest['own'][pair+'/'+policy]
        by_policy[policy] = Counter(int(r['local_query']) for r in rows)
        assert len({r['uid'] for r in rows}) == len(rows)
        for row in rows:
            item = {k: row[k] for k in ('uid', 'local_query', 'local_track', 'row_A', 'row_B')}
            item['case'] = manifest['cases'][row['uid']]
            if item['uid'] in actions:
                assert actions[item['uid']] == item
            actions[item['uid']] = item
    assert by_policy['DESIGN'] == by_policy['UNIFORM']
    return actions, dict(by_policy['DESIGN'])


def group_of(action):
    return int(action['local_query']), int(action['local_track'])


def reserve_probes(actions, quotas, fraction=.1, rounds=3):
    """Fix fresh query/track groups without examining teacher responses.

    Reserving a whole group must leave enough other actions for the declared
    query budget. Every query retains at least one ordinary training request.
    Very small libraries may support fewer fresh probes; absence is recorded,
    and no semantic reward is manufactured for an empty probe round.
    """
    candidates = defaultdict(list)
    for uid, action in sorted(actions.items()):
        candidates[int(action['local_query'])].append(uid)
    wanted = min(sum(quotas.values())//3, max(rounds, round(sum(quotas.values())*fraction)))
    order = sorted(actions, key=lambda uid: hashlib.sha256(('rsi-probe-v1:'+uid).encode()).digest())
    reserved, groups, counts = [], set(), Counter()
    for uid in order:
        action = actions[uid]
        qi, _ = group_of(action)
        group = group_of(action)
        if len(reserved) == wanted:
            break
        if group in groups or counts[qi]+1 >= quotas[qi]:
            continue
        proposed = groups | {group}
        available = sum(group_of(actions[u]) not in proposed for u in candidates[qi])
        if available < quotas[qi]-counts[qi]-1:
            continue
        reserved.append(uid)
        groups.add(group)
        counts[qi] += 1
    probes = [reserved[i::rounds] for i in range(rounds)]
    schedule = []
    for stage in range(rounds):
        schedule.append({qi: (quotas[qi]-counts[qi])//rounds+
                         int(stage < (quotas[qi]-counts[qi]) % rounds) for qi in quotas})
    for qi, budget in quotas.items():
        assert sum(s[qi] for s in schedule)+counts[qi] == budget
        assert schedule[0][qi] >= 1
    return dict(probes=probes, training_schedule=schedule, groups=groups,
                requested_probe_count=wanted, actual_probe_count=len(reserved),
                fresh_probe_rounds=sum(bool(p) for p in probes))


class TeacherReplayBroker:
    """Private finite teacher library, exposing and charging requested UIDs.

    Loading an answer library is research preparation, not a new model call.
    The selector receives only ``actions``. Answers are returned by request;
    probe feedback is held out of the current solver update by the caller.
    """
    def __init__(self, collection, pair, video, actions, quotas, reservation):
        self.collection, self.pair, self.video = collection, pair, video
        self.actions, self.quotas, self.reservation = actions, quotas, reservation
        self._answers = None
        self._served = {}
        self.sequence = []
        self.counts = Counter()

    def _load(self):
        from scripts.a100_observation_design_collect import TEACHER_SHA
        answers = {}
        for policy in ('DESIGN', 'UNIFORM'):
            path = self.collection/'policy_answers'/self.pair/self.video/(policy+'_answers.json')
            doc = json.loads(path.read_text())
            assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
            assert doc['teacher_weight_sha256'] == TEACHER_SHA
            for row in doc['cases']:
                uid = row['uid']
                if uid in answers:
                    assert answers[uid]['joint_target'] == row['joint_target']
                    assert answers[uid]['shared_binary_target'] == row['shared_binary_target']
                answers[uid] = row
        assert set(answers) == set(self.actions)
        self._answers = answers

    def request(self, uids, *, role, stage):
        assert role in ('train', 'probe') and 0 <= stage < 3
        if self._answers is None:
            self._load()
        result = []
        for uid in uids:
            action = self.actions[uid]
            qi, _ = group_of(action)
            if role == 'probe':
                assert uid in self.reservation['probes'][stage]
            else:
                assert group_of(action) not in self.reservation['groups']
            repeat = uid in self._served
            if not repeat:
                if self.counts[qi] >= self.quotas[qi]:
                    raise ValueError('Teacher query budget exhausted: '+str(qi))
                self.counts[qi] += 1
                # Only requested metadata/targets leave the broker. Raw
                # model tokens or unrelated library targets do not.
                row = self._answers[uid]
                self._served[uid] = {**action,
                    'joint_target': list(row['joint_target']),
                    'shared_binary_target': list(row['shared_binary_target'])}
            self.sequence.append(dict(uid=uid, role=role, stage=stage, repeat=repeat))
            result.append(self._served[uid])
        return result

    def receipt(self, complete=False):
        if complete:
            assert dict(self.counts) == self.quotas
        unique = list(self._served.values())
        return dict(unique_teacher_bundles=len(unique), teacher_budget=sum(self.quotas.values()),
                    per_query_requested=dict(self.counts), per_query_budget=self.quotas,
                    probe_bundles=sum(r['role'] == 'probe' and not r['repeat'] for r in self.sequence),
                    logical_requests=len(self.sequence),
                    original_joint_channel_examples=sum(1 if r['row_A'] == r['row_B'] else 2 for r in unique),
                    new_teacher_GPU_examples=0, prepaid_research_library=True,
                    not_single_method_cold_cost=True, unrequested_answers_delivered=0,
                    selected_UID_SHA256=hashlib.sha256('\n'.join(sorted(self._served)).encode()).hexdigest())


class ProgressChallenger(nn.Module):
    """A learned observation policy, separate from the RMOT solver."""
    def __init__(self, dimension, width=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def exploration_log_probabilities(scores, epsilon=.2):
    assert 0 < epsilon < 1 and scores.ndim == 1 and len(scores)
    log_policy = scores.double().log_softmax(0)
    return torch.logaddexp(log_policy+math.log1p(-epsilon),
                           torch.full_like(log_policy, math.log(epsilon/len(scores))))


def select_round(actor, features, actions, quotas, reservation, served, stage, generator):
    """Sample a fixed query quota without replacement and with track coverage.

    At fixed state this is a proper finite stochastic policy. The selected
    log mixture probabilities, including exploration, define its score
    function; target replies are not used until after the whole batch.
    """
    order = sorted(actions)
    values = torch.as_tensor(features, dtype=next(actor.parameters()).dtype,
                             device=next(actor.parameters()).device)
    assert values.shape[0] == len(order)
    scores = actor(values)
    index = {uid: i for i, uid in enumerate(order)}
    used = set(served)
    seen = {group_of(actions[uid]) for uid in used}
    chosen, log_probabilities = [], []
    for qi in sorted(quotas):
        for _ in range(reservation['training_schedule'][stage][qi]):
            available = [uid for uid in order if int(actions[uid]['local_query']) == qi
                         and uid not in used and group_of(actions[uid]) not in reservation['groups']]
            fresh = [uid for uid in available if group_of(actions[uid]) not in seen]
            eligible = fresh or available
            assert eligible, (qi, stage, 'Insufficient unlabelled action library')
            logp = exploration_log_probabilities(scores[[index[uid] for uid in eligible]])
            draw = int(torch.multinomial(logp.detach().exp().cpu(), 1, generator=generator))
            uid = eligible[draw]
            chosen.append(uid)
            log_probabilities.append(logp[draw])
            used.add(uid)
            seen.add(group_of(actions[uid]))
    return chosen, (torch.stack(log_probabilities).mean() if log_probabilities else scores.sum()*0)


def update_challenger(actor, optimizer, log_probability, before, after, baseline, *, frozen=False):
    """One myopic policy-gradient step on fresh, paid teacher-probe progress."""
    reward = math.tanh(before-after) if before is not None else 0.
    advantage = reward-baseline
    before_parameters = torch.cat([p.detach().flatten().cpu().double() for p in actor.parameters()])
    if not frozen and before is not None:
        optimizer.zero_grad(set_to_none=True)
        loss = -advantage*log_probability
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.)
        optimizer.step()
    after_parameters = torch.cat([p.detach().flatten().cpu().double() for p in actor.parameters()])
    return dict(reward=reward, baseline_before=baseline,
                baseline_after=.9*baseline+.1*reward if before is not None else baseline,
                teacher_probe_CE_before=before, teacher_probe_CE_after=after,
                actor_parameter_L2_change=float((after_parameters-before_parameters).norm()),
                frozen_actor=frozen, fresh_teacher_probe=before is not None,
                reward_is_teacher_risk_proxy_not_RMOT_truth=True)


def challenger_features(log_probability, arrays, actions, served):
    """Bounded student/visual/exposure features, with no teacher targets."""
    order = sorted(actions)
    q = np.exp(log_probability)
    assert q.shape == (len(order), 4)
    assert np.isfinite(q).all() and np.allclose(q.sum(-1), 1.)
    entropy = -(q*np.log(q.clip(min=1e-300))).sum(-1, keepdims=True)/math.log(4.)
    first, second = arrays['first'], arrays['second']
    a, b = first[:, :768], second[:, :768]
    cosine = (a*b).sum(-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)).clip(min=1e-12)
    colour = np.minimum(np.square(first[:, 768:1080]-second[:, 768:1080]).sum(-1)/6, 1.)
    geometry = np.abs(first[:, -6:-2]-second[:, -6:-2]).clip(0, 1)
    relative = np.abs(first[:, -18:-6]-second[:, -18:-6]).mean(-1).clip(0, 1)
    exposure = Counter(group_of(actions[u]) for u in served)
    query_exposure = Counter(int(actions[u]['local_query']) for u in served)
    repeats = np.asarray([min(exposure[group_of(actions[u])], 9)/9 for u in order])
    counts = np.asarray([min(query_exposure[int(actions[u]['local_query'])], 64)/64 for u in order])
    views = np.eye(3)[[('CROP', 'FRAME_BOX', 'CLIP').index(actions[u]['case']['view']) for u in order]]
    gap = np.abs(first[:, -2]-second[:, -2])
    result = np.c_[q, entropy, ((1-cosine)/2).clip(0, 1), colour,
                   geometry, relative, repeats, counts, views, gap]
    assert np.isfinite(result).all()
    return result.astype('float32')
