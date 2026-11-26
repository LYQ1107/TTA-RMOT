"""Paid external-evidence observer/solver loop for a small primary pilot.

Independent means the current probe labels are withheld from this update,
not that correlated video frames become independent statistical samples.
Teacher progress is a proxy; neither agreement nor confidence is a true label.
"""
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch import nn
from flex_uda.active.factor_evidence import MODES, FactorStudent
from flex_uda.active.native_binary_inputs import coarse_context
from flex_uda.active.rsi_curriculum import exploration_log_probabilities
from flex_uda.active.rsi_policy_distribution import trust_update


def uid(case):
    keys = ('target', 'video_id', 'query_id', 'query_text', 'track_id', 'frame_id', 'view', 'factor_mode')
    return hashlib.sha256(json.dumps([case[k] for k in keys], ensure_ascii=False).encode()).hexdigest()


class Observer(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(8, 1, bias=False).double()
        nn.init.zeros_(self.linear.weight)

    def forward(self, values):
        # Soft coverage bonus, never the old hard fresh-object priority.
        return self.linear(values).flatten() + math.log(2)*values[:, 4]


class Inputs:
    def __init__(self, video):
        self.video = video
        raw = video.raw_bags[:, video.present].astype('float64')
        self.mean = raw.mean((0, 1)); self.std = raw.std((0, 1)).clip(min=.01)
        self.coarse = coarse_context(video.raw_bags, np.broadcast_to(video.present, video.raw_bags.shape[:-1]))

    def arrays(self, cases):
        v = self.video; q = np.asarray([r['local_query'] for r in cases]); t = np.asarray([r['local_track'] for r in cases])
        rows = np.asarray([v.positions[int(r['frame_id']), int(r['track_id'])] for r in cases])
        present = v.present[t].copy()
        bags = ((v.raw_bags[q, t]-self.mean)/self.std).astype('float32'); bags[~present] = 0
        current = ((v.dense_rows(q, rows)-self.mean)/self.std).astype('float32')
        coarse = ((self.coarse[q, t]-self.mean)/self.std).astype('float32')
        return dict(bags=bags, present=present, current=current, coarse=coarse)


def tensor_data(arrays, device):
    return {k:torch.as_tensor(v, device=device) for k, v in arrays.items()}


def teacher_floor(arrays, roles, target):
    # Exact common-input conflicts are an attainable floor, not fit failures.
    packed = np.ascontiguousarray(np.c_[arrays['bags'].reshape(len(roles), -1), arrays['present'],
        arrays['current'], arrays['coarse'], roles], dtype='float32')
    _, inverse = np.unique(packed.view(np.dtype((np.void, packed.shape[1]*4))).ravel(), return_inverse=True)
    count = np.bincount(inverse); average = np.bincount(inverse, weights=target)/count
    ent = entropy(average)[inverse]
    return np.asarray([ent[roles == r].mean() if np.any(roles == r) else np.nan for r in range(3)])


def entropy(p):
    p = np.asarray(p, dtype='float64').clip(1e-12, 1-1e-12)
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def fit(model, inputs, paid, *, steps, device, hard=False, whole_only=False):
    started = time.perf_counter()
    selected = [r for r in paid if not whole_only or r['factor_mode'] == 'WHOLE']
    assert selected
    arrays = inputs.arrays(selected); data = tensor_data(arrays, device)
    roles_np = np.asarray([MODES.index(r['factor_mode']) for r in selected])
    target_np = np.asarray([r['yes_probability'] for r in selected], dtype='float64')
    if hard: target_np = (target_np >= .5).astype('float64')
    floor = teacher_floor(arrays, roles_np, target_np)
    roles = torch.as_tensor(roles_np, device=device); target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for begin in range(0, len(selected), 512):
            block = {k:v[begin:begin+512] for k, v in data.items()}
            logits = model(**block).gather(1, roles[begin:begin+512, None]).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[begin:begin+512], reduction='sum')/len(selected)
            loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(**{k:v[b:b+512] for k, v in data.items()})
                            for b in range(0, len(selected), 512)]).gather(1, roles[:, None]).flatten().double()
        y = torch.as_tensor(target_np, device=device)
        risks = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none').cpu().numpy()
    per_mode = {MODES[r]:dict(teacher_CE=float(risks[roles_np == r].mean()),
        attainable_entropy=float(floor[r]), excess_KL=float(risks[roles_np == r].mean()-floor[r]),
        rows=int(sum(roles_np == r))) for r in range(3) if np.any(roles_np == r)}
    return dict(steps=steps, labels=len(selected), roles=per_mode, hard=hard, whole_only=whole_only,
                wall_seconds=time.perf_counter()-started,
                seed=0, manual_GT_selection=False, risk_units='proper teacher KL in nats')


def make_pool(video, quotas):
    from flex_uda.active.acquisition import route
    pool = {}
    for qi, query in enumerate(video.queries):
        assert qi in quotas
        for ti, track in enumerate(video.tracks):
            for si in np.flatnonzero(video.present[ti]):
                row = int(video.sample_rows[ti, si]); frame = int(video.obs[row, 0])
                for mode in MODES:
                    case = dict(target=video.target, video_id=video.video, query_id=query['expression_id'],
                        query_text=query['query_text'], track_id=int(track), frame_id=frame,
                        view='CROP' if mode == 'APPEARANCE' else 'FRAME_BOX' if mode == 'STATE' else route(query['query_text']),
                        factor_mode=mode, local_query=qi, local_track=ti)
                    pool[uid(case)] = case
    return pool


def probes_and_seeds(video, original, quotas, pool):
    v = video; byq = {}
    for r in original: byq.setdefault(int(r['local_query']), r)
    frames, counts = np.unique(v.obs[:, 0], return_counts=True)
    middle = np.median(frames)
    frame = min(zip(frames, counts), key=lambda x:(-x[1], abs(x[0]-middle), x[0]))[0]
    ids = np.flatnonzero(v.obs[:, 0] == frame)
    assert len(ids) >= 2
    ids = ids[np.argsort(v.obs[ids, 1], kind='stable')][[0, -1]]
    chosen = sorted((q for q in quotas if quotas[q] >= 3),
                    key=lambda q:(-quotas[q], v.queries[q]['expression_id']))[:3]
    assert len(chosen) == 3, 'This registered pilot needs three paid probe pairs'
    probes = []
    for q in chosen:
        cases = []
        for row in ids:
            track = int(v.obs[row, 1]); ti = int(v.columns[row]); query = v.queries[q]
            cases.append(dict(target=v.target, video_id=v.video, query_id=query['expression_id'],
                query_text=query['query_text'], track_id=track, frame_id=int(frame), view='FRAME_BOX',
                factor_mode='WHOLE', local_query=q, local_track=ti))
        probes.append(cases)
    reserved = {(r['local_query'], r['track_id'], r['frame_id']) for pair in probes for r in pair}
    seeds = []
    for q in sorted(quotas):
        if q in chosen: continue
        case = dict(byq[q], factor_mode='WHOLE'); seeds.append(case)
    remaining = {q:quotas[q]-(2 if q in chosen else 1) for q in quotas}
    assert min(remaining.values()) >= 0
    return probes, seeds, remaining, reserved


def choose(actor, model, inputs, pool, quotas, served, reserved, generator):
    used = {uid(r) for r in served}; seen = {(r['local_query'], r['track_id']) for r in served}
    selected = []; contexts = []; decisions = []; trace = []; offset = 0
    device = next(model.parameters()).device
    for q, number in sorted(quotas.items()):
        if not number: continue
        eligible = [(u, r) for u, r in sorted(pool.items()) if r['local_query'] == q and u not in used
                    and (q, r['track_id'], r['frame_id']) not in reserved]
        assert len(eligible) >= number
        cases = [r for _, r in eligible]
        with torch.no_grad():
            probability = model(**tensor_data(inputs.arrays(cases), device)).sigmoid().cpu().numpy()
        base = np.column_stack((entropy(probability[:, 0]), entropy(probability[:, 1]), entropy(probability[:, 2]),
            np.abs(probability[:, 0]-probability[:, 1]*probability[:, 2]), np.zeros(len(cases)),
            *[np.asarray([r['factor_mode'] == m for r in cases], float) for m in MODES]))
        available = list(range(len(cases)))
        for _ in range(number):
            feature = base[available].copy()
            feature[:, 4] = [float((q, cases[i]['track_id']) not in seen) for i in available]
            values = torch.as_tensor(feature, dtype=torch.float64)
            with torch.no_grad():
                logp = exploration_log_probabilities(actor(values))
                frozen = exploration_log_probabilities(math.log(2)*values[:, 4])
            draw = float(generator.random()); at = min(int(np.searchsorted(np.cumsum(logp.exp().numpy()), draw)), len(available)-1)
            frozen_at = min(int(np.searchsorted(np.cumsum(frozen.exp().numpy()), draw)), len(available)-1)
            index = available[at]; case = cases[index]; selected.append(case)
            contexts.append(values); decisions.append((torch.arange(offset, offset+len(values)), at, logp))
            trace.append(dict(uid=uid(case), query=q, factor_mode=case['factor_mode'], probability=float(logp[at].exp()),
                one_step_frozen_choice_uid=uid(cases[available[frozen_at]]),
                counterfactual_is_conditional_on_actual_history_not_a_quality_ablation=True))
            offset += len(values); seen.add((q, case['track_id'])); available.pop(at)
    values = torch.cat(contexts) if contexts else torch.empty((0, 8), dtype=torch.float64)
    actor._distribution_trace = values, decisions
    return selected, trace


def discrimination_credit(before, after, answers):
    p = np.asarray([r['yes_probability'] for r in answers], dtype='float64').clip(1e-6, 1-1e-6)
    assert len(p) == 2
    delta = float(np.log(p[0]/(1-p[0]))-np.log(p[1]/(1-p[1])))
    target = 1/(1+np.exp(-delta))
    def risk(value):
        difference = float(value[0]-value[1])
        return float(np.logaddexp(0, difference)-target*difference)
    first, last = risk(before), risk(after)
    information = float(entropy(p.mean())-entropy(p).mean())
    reward = max(0., information)/math.log(2)*math.tanh(first-last)
    return dict(before=first, after=last, reward=reward, teacher_JS=information,
        teacher_pair_target=target, target_is_independent_binary_conditional_model_not_joint_truth=True,
        common_student_logit_shift_cannot_change_reward=True, reward_is_not_GT_or_HOTA=True)
