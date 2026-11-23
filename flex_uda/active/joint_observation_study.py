"""Matched, label-free inputs and fitting for the joint-observation study.

This module never evaluates RMOT predictions. Teacher opinions are projected
onto the common visual input, not interpreted as ground-truth identities.
"""
from collections import defaultdict
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flex_uda.active.conditional_joint_observation import ConditionalJointObservationHead
from flex_uda.active.joint_observation import JointObservationHead
from flex_uda.active.joint_observation_fit import change_normalization, probabilities


def video_records(video, collection):
    """Read this video's own-policy answers; no other video's labels or QA."""
    from scripts.a100_joint_observation_collect import teacher_pool

    nq, nt, dimension = len(video.queries), len(video.tracks), video.dimension
    pools = teacher_pool(video)
    pool_rows = np.full((nt, 3), -1, dtype=int)
    pool_present = np.zeros((nt, 3), dtype=bool)
    pool_values = np.zeros((nq, nt, 3, dimension), dtype='float32')
    for ti, rows in enumerate(pools):
        pool_rows[ti, :len(rows)] = rows
        pool_present[ti, :len(rows)] = True
    ti, slot = np.nonzero(pool_present)
    for qi in range(nq):
        pool_values[qi, ti, slot] = video.dense_rows(
            np.full(len(ti), qi, dtype=int), pool_rows[ti, slot])

    # All ordinary learners may summarize the same context plus reference
    # pool. Real duplicate observations count once, including short tracks.
    coarse = np.zeros((nq, nt, dimension), dtype='float32')
    norm_sum = np.zeros(dimension)
    norm_square = np.zeros(dimension)
    norm_rows = 0
    for ti in range(nt):
        context = video.sample_rows[ti, video.present[ti]]
        extra = [s for s, row in enumerate(pools[ti]) if row not in set(context)]
        values = np.concatenate((video.raw_bags[:, ti, video.present[ti]],
                                 pool_values[:, ti, extra]), axis=1).astype('float64')
        assert values.shape[1] == len(set(context) | set(pools[ti]))
        coarse[:, ti] = values.mean(1)
        # Preserve the original sqrt(mean histogram) descriptor, with one
        # denominator for each observed body part, not mean(sqrt histogram).
        hist = np.square(values[..., 768:1080]).reshape(nq, values.shape[1], 3, 104)
        flags = values[..., 1080:1083]
        coarse[:, ti, 768:1080] = np.sqrt(
            hist.sum(1)/np.maximum(flags.sum(1)[..., None], 1)).reshape(nq, 312)
        norm_sum += values.sum((0, 1))
        norm_square += np.square(values).sum((0, 1))
        norm_rows += values.shape[0]*values.shape[1]

    records = {}
    for policy in ('DESIGN', 'UNIFORM'):
        path = collection/'policy_answers'/video.pair/video.video/(policy+'_answers.json')
        doc = json.loads(path.read_text())
        assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
        grouped, endpoint = defaultdict(list), defaultdict(list)
        seen = set()
        for row in doc['cases']:
            assert row['uid'] not in seen
            seen.add(row['uid'])
            qi, ti, ra, rb = (int(row[k]) for k in ('local_query', 'local_track', 'row_A', 'row_B'))
            case = row['case']
            assert case['query_id'] == video.queries[qi]['expression_id']
            assert case['query_text'] == video.queries[qi]['query_text']
            assert int(case['track_id']) == int(video.tracks[ti])
            assert tuple(video.obs[ra]) == (case['frame_A'], case['track_id'])
            assert tuple(video.obs[rb]) == (case['frame_B'], case['track_id'])
            assert ra in pools[ti] and rb in pools[ti] and ra <= rb
            joint = np.asarray(row['joint_target'], dtype='float64')
            binary = np.asarray(row['shared_binary_target'], dtype='float64')
            assert joint.shape == (4,) and binary.shape == (2,)
            assert np.isfinite(joint).all() and np.isfinite(binary).all()
            assert np.all(joint >= 0) and np.all((binary >= 0) & (binary <= 1))
            joint /= joint.sum()
            a, b = binary
            binary_joint = np.asarray([a*(1-b), (1-a)*b, a*b, (1-a)*(1-b)])
            if ra == rb:
                assert np.allclose(joint, [0, 0, a, 1-a]) and a == b
                binary_joint = joint.copy()
            grouped[qi, ti, ra, rb].append((joint, binary_joint))
            endpoint[qi, ti, ra].append(joint[0]+joint[2])
            if rb != ra:
                endpoint[qi, ti, rb].append(joint[1]+joint[2])
        keys = np.asarray(sorted(grouped), dtype=int)
        assert set(keys[:, 0]) == set(range(nq))
        qi, ti, ra, rb = keys.T
        joint = np.stack([np.mean([p[0] for p in grouped[tuple(k)]], axis=0) for k in keys])
        binary = np.stack([np.mean([p[1] for p in grouped[tuple(k)]], axis=0) for k in keys])
        # View is not an input to the learned head. Distinct requested views
        # at identical observation pairs supply one common arithmetic mixture
        # target for EVERY learner; repeating a view does not invent examples.
        bags = defaultdict(list)
        for (q, t, _), opinions in endpoint.items():
            bags[q, t].append(float(np.mean(opinions)))
        coarse_keys = np.asarray(sorted(bags), dtype=int)
        records[policy] = dict(
            anchors=video.raw_bags[qi, ti], present=video.present[ti],
            first=video.dense_rows(qi, ra), second=video.dense_rows(qi, rb),
            joint_target=joint, binary_target=binary,
            coarse_inputs=coarse[coarse_keys[:, 0], coarse_keys[:, 1]],
            coarse_target=np.asarray([np.mean(bags[tuple(k)]) for k in coarse_keys]),
            report=dict(policy=policy, requested_bundles=len(seen),
                        unique_visual_pairs=len(keys), coarse_bags=len(coarse_keys),
                        mixed_view_pairs=sum(len(v) > 1 for v in grouped.values()),
                        same_visual_input_mixture_shared_by_all=True,
                        coarse_equal_weight_unique_observations=True))
    return dict(records=records, pools=pools, pool_values=pool_values,
                pool_present=pool_present, coarse=coarse,
                norm_sum=norm_sum, norm_square=norm_square, norm_rows=norm_rows)


def prefix_arrays(records, mean, std):
    data = {key: np.concatenate([r[key] for r in records]) for key in
            ('anchors', 'present', 'first', 'second', 'joint_target', 'binary_target',
             'coarse_inputs', 'coarse_target')}
    for key in ('anchors', 'first', 'second', 'coarse_inputs'):
        data[key] = ((data[key]-mean)/std).astype('float32')
    data['anchors'][~data['present']] = 0
    return data


def make_model(recipe, dimension, width):
    if recipe['family'] == 'structured':
        return ConditionalJointObservationHead(
            dimension, width=width,
            conditional_estimation=recipe['conditional'],
            object_reference=recipe['reference'])
    if recipe['family'] == 'generic':
        return JointObservationHead(dimension, width=width, generic=True)
    assert recipe['family'] == 'coarse'
    model = nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, 1))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    return model


def coalesce_exact_inputs(arrays, recipe):
    """Preserve CE exactly; compute its attainable identical-input floor.

    Opposing opinions attached to byte-identical model inputs cannot be fit
    individually by any deterministic learner. Their weighted mean is the
    sufficient statistic for proper cross entropy, not a weakened baseline.
    """
    coarse = recipe['family'] == 'coarse'
    keys = ('coarse_inputs',) if coarse else ('anchors', 'present', 'first', 'second')
    if coarse:
        raw = arrays['coarse_target']
        y = (raw >= .5).astype('float64') if recipe['hard'] else (
            np.clip(raw, .02, .98) if recipe['confidence'] else raw)
        target = np.c_[y, 1-y]
        weights = 1+3*np.abs(2*raw-1) if recipe['confidence'] else np.ones(len(y))
    else:
        target = arrays['binary_target' if recipe.get('binary') else 'joint_target'].copy()
        weights = np.ones(len(target))
    target /= target.sum(-1, keepdims=True)
    lookup, owners, inverse = {}, [], []
    for row in range(len(target)):
        digest = hashlib.sha256()
        for key in keys:
            digest.update(np.ascontiguousarray(arrays[key][row]).tobytes())
        signature = digest.digest()
        if signature not in lookup:
            lookup[signature] = len(owners)
            owners.append(row)
        inverse.append(lookup[signature])
    inverse = np.asarray(inverse)
    mass = np.bincount(inverse, weights=weights)
    projected = np.stack([np.bincount(inverse, weights=weights*target[:, j])
                          for j in range(target.shape[-1])], -1)/mass[:, None]
    selected = {key: arrays[key][owners] for key in keys}
    selected.update(target=projected, weight=mass)
    return selected, dict(original_risk_rows=len(target), exact_input_groups=len(owners),
                          identical_input_risk_preserved=True)


def model_log_probabilities(model, data, coarse=False):
    if coarse:
        z = model(data['coarse_inputs']).squeeze(-1).double()
        return torch.stack((F.logsigmoid(z), F.logsigmoid(-z)), -1)
    return probabilities(model, data)


def teacher_risk(logp, target, weight, kind):
    if kind == 'marginal':
        target = torch.stack((torch.stack((target[:, 0]+target[:, 2], target[:, 1]+target[:, 3]), -1),
                              torch.stack((target[:, 1]+target[:, 2], target[:, 0]+target[:, 3]), -1)), -2)
        logp = torch.stack((torch.stack((torch.logsumexp(logp[:, (0, 2)], -1), torch.logsumexp(logp[:, (1, 3)], -1)), -1),
                            torch.stack((torch.logsumexp(logp[:, (1, 2)], -1), torch.logsumexp(logp[:, (0, 3)], -1)), -1)), -2)
    ce = -(target*logp).sum(-1)
    entropy = -torch.xlogy(target, target).sum(-1)
    if kind == 'marginal':
        ce, entropy = ce.mean(-1), entropy.mean(-1)
    return (ce*weight).sum()/weight.sum(), (entropy*weight).sum()/weight.sum()


def fit_prefix(arrays, recipe, mean, std, previous=None, device='cuda:0'):
    """Only teacher risk controls ordinary optimization/capacity completion."""
    started = time.perf_counter()
    dimension = len(mean)
    selected, projection = coalesce_exact_inputs(arrays, recipe)
    data = {k: torch.as_tensor(v, device=device) for k, v in selected.items()}
    coarse = recipe['family'] == 'coarse'
    kind = recipe.get('risk', 'joint')
    ordinary = recipe['family'] != 'structured'
    initial_width = previous['width'] if previous else (128 if recipe['family'] == 'generic' else 64)
    widths = [initial_width]+([w for w in (128, 256, 512) if w > initial_width] if ordinary else [])
    minimum = 400 if previous else 1000
    trace, attempts = [], []
    early = None
    final = None
    for width_index, width in enumerate(widths):
        torch.manual_seed(0)
        model = make_model(recipe, dimension, width).to(device)
        if width_index == 0 and previous is not None:
            assert previous['recipe'] == recipe
            model.load_state_dict(previous['model'])
            if recipe['family'] == 'structured':
                model.renormalize(previous['mean'], previous['std'], mean, std)
            elif coarse:
                with torch.no_grad():
                    layer = model[0]
                    before = layer.weight.clone()
                    ratio = torch.as_tensor(std/previous['std'], device=device, dtype=before.dtype)
                    delta = torch.as_tensor((mean-previous['mean'])/previous['std'], device=device, dtype=before.dtype)
                    layer.weight.mul_(ratio)
                    layer.bias.add_(before@delta)
            else:
                change_normalization(model, previous['mean'], previous['std'], mean, std)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for step in range(1, 4001):
            optimizer.zero_grad(set_to_none=True)
            ce, floor = teacher_risk(model_log_probabilities(model, data, coarse), data['target'], data['weight'], kind)
            assert torch.isfinite(ce)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 400:
                early = copy.deepcopy(model).eval()
            if step >= minimum and (step % 250 == 0 or step == 400):
                with torch.no_grad():
                    c, h = teacher_risk(model_log_probabilities(model, data, coarse), data['target'], data['weight'], kind)
                metrics = dict(width=width, steps=step, teacher_CE=float(c), attainable_exact_input_entropy=float(h),
                               excess_KL=float(c-h), optimizer='Adam')
                trace.append(metrics)
                if metrics['excess_KL'] <= .005:
                    break
                # Bounded structural models may have inconsistent teacher
                # pair orderings. Do not spend 16k steps chasing an impossible
                # entropy floor, or report them as adequate ordinary KD.
                if not ordinary and len(trace) >= 4 and step >= minimum+750:
                    recent = [r['excess_KL'] for r in trace[-4:]]
                    if max(recent)-min(recent) <= 1e-5:
                        break
        if ordinary and metrics['excess_KL'] > .005:
            # Full-batch continuation, inspected as deployed float32. No GT,
            # seed search, or favorable HOTA checkpoint enters this decision.
            model.double()
            double_data = {k: (v if v.dtype == torch.bool else v.double()) for k, v in data.items()}
            opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=100, max_eval=200,
                                  history_size=10, tolerance_grad=1e-10,
                                  tolerance_change=1e-14, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True)
                loss, _ = teacher_risk(model_log_probabilities(model, double_data, coarse),
                                       double_data['target'], double_data['weight'], kind)
                loss.backward()
                return loss
            for block in range(5):
                opt.step(closure)
                deployed = make_model(recipe, dimension, width).to(device)
                deployed.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
                with torch.no_grad():
                    c, h = teacher_risk(model_log_probabilities(deployed, data, coarse), data['target'], data['weight'], kind)
                metrics = dict(width=width, LBFGS_blocks=block+1, teacher_CE=float(c),
                               attainable_exact_input_entropy=float(h), excess_KL=float(c-h),
                               optimizer='LBFGS_float64_checked_in_float32')
                attempts.append(metrics)
                final = deployed
                if metrics['excess_KL'] <= .005:
                    break
            model = final
        if not ordinary or metrics['excess_KL'] <= .005:
            break
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    receipt = dict(recipe=recipe, width=width, minimum_Adam_steps=minimum,
                   early_step=400, actual_final_Adam_steps=step, trace=trace,
                   capacity_and_LBFGS_attempts=attempts, metrics=metrics,
                   adequate_ordinary_required=ordinary, teacher_fit_gate_passed=metrics['excess_KL'] <= .005,
                   structural_residual_not_an_ordinary_fit_pass=not ordinary,
                   neural_seed=0, prefix_warm_start=previous is not None,
                   target_GT_or_HOTA_used=False, **projection,
                   wall_seconds=time.perf_counter()-started)
    if ordinary and not receipt['teacher_fit_gate_passed']:
        raise RuntimeError('Ordinary teacher fit needs numerical repair; no quality comparison: '+str(receipt))
    checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      width=width, recipe=recipe, mean=mean.copy(), std=std.copy(), receipt=receipt)
    assert early is not None
    return model.eval(), early, checkpoint, receipt
