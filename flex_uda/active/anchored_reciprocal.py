"""Teacher-anchor visual challenges and bounded-influence reciprocal learning.

The external teacher is noisy guidance, not manual truth. The two students
exchange positive sample weights, and every query/role retains its original
total supervision mass. A path edge cover ensures that regularization reaches
the paid image anchors, rather than only unrelated unlabeled frames.
"""
from collections import defaultdict
import math
import time

import numpy as np
import torch
from torch import nn

from flex_uda.active.factor_evidence import MODES
from flex_uda.active.reciprocal_memory import curriculum, teacher_risks, tensor_data


def peer_weights(kl, groups):
    """Stop-gradient peer weights; each query/role keeps its exact row mass."""
    weight = .2 + .8/(1+kl.detach().clamp(min=0)/math.log(2))
    counts = torch.bincount(groups).to(weight.dtype)
    totals = torch.zeros_like(counts).index_add_(0, groups, weight)
    return weight * counts[groups]/totals[groups]


def anchor_edge_cover(inputs, paid):
    """Minimum-cardinality edge cover of marked vertices on disjoint paths.

Maximum matching pairs consecutive marked vertices within each marked run;
each remaining nonisolated marked vertex takes an incident edge. Minimum
cardinality refers to topology only, not information, semantics or HOTA.
"""
    video = inputs.video
    forward = {int(row): i for i, row in enumerate(inputs.left)}
    backward = {int(row): i for i, row in enumerate(inputs.right)}
    marked = {(int(r['local_query']), video.positions[int(r['frame_id']), int(r['track_id'])]) for r in paid}
    by_track = defaultdict(list)
    for query, row in marked:
        by_track[query, int(video.columns[row])].append(row)
    selected = set()
    isolated = []
    matching_edges = 0
    for (query, track), rows in sorted(by_track.items()):
        rows.sort(key=lambda row: (video.obs[row, 0], row))
        index = 0
        while index < len(rows):
            row = rows[index]
            edge = forward.get(row)
            if index+1 < len(rows) and edge is not None and inputs.right[edge] == rows[index+1]:
                selected.add(query*len(inputs.left)+edge)
                matching_edges += 1
                index += 2
                continue
            incident = [e for e in (forward.get(row), backward.get(row)) if e is not None]
            if not incident:
                isolated.append((query, int(row)))
            else:
                # Observable image support only, no teacher answer or model score.
                best = max(incident, key=lambda e: (inputs.visual[e], -e))
                selected.add(query*len(inputs.left)+best)
            index += 1
    touched = {(int(e//len(inputs.left)), int(row)) for e in selected
               for row in (inputs.left[e % len(inputs.left)], inputs.right[e % len(inputs.left)])}
    assert marked-touched == set(isolated)
    assert len(selected) == len(marked)-len(isolated)-matching_edges
    return sorted(selected), dict(unique_paid_image_groups=len(marked), isolated_paid_image_groups=len(isolated),
                                  covered_paid_image_groups=len(marked & touched),
                                  minimum_mandatory_edges=len(selected), matched_marked_pairs=matching_edges,
                                  minimum_is_topological_not_semantic=True)


def challenge(pair, inputs, paid, weights, generator, device):
    mandatory, receipt = anchor_edge_cover(inputs, paid)
    _, old = curriculum(pair, inputs, weights, generator, device)
    ids = np.asarray(sorted(set(mandatory) | set(old['selected_edge_ids'])), dtype=int)
    left, right, weight = inputs.edge_data(ids)
    receipt.update(selected_edge_ids=ids.tolist(), selected_edges=len(ids),
                   background_disagreement_edges=old['selected_edges'],
                   epsilon=old['epsilon'], mean_selected_role_weights=weight.mean(0).tolist(),
                   paid_teacher_answers_not_used_for_edge_cover=True,
                   no_new_teacher_queries=True)
    return (left, right, weight), receipt


def train(pair, inputs, paid, *, device, steps, edges, weights):
    start = time.perf_counter()
    arrays = inputs.arrays(paid)
    data = tensor_data(arrays, device)
    roles = torch.as_tensor([MODES.index(r['factor_mode']) for r in paid], device=device)
    target = torch.as_tensor([r['yes_probability'] for r in paid], dtype=torch.float32, device=device)
    distinct = {key: i for i, key in enumerate(sorted({(r['local_query'], r['factor_mode']) for r in paid}))}
    groups = torch.as_tensor([distinct[r['local_query'], r['factor_mode']] for r in paid], device=device)
    entropy = -(target.clamp(min=1e-12)*target.clamp(min=1e-12).log()
                +(1-target).clamp(min=1e-12)*(1-target).clamp(min=1e-12).log())
    a, b, edge_weight = edges
    left, right = tensor_data(a, device), tensor_data(b, device)
    edge_weight = torch.as_tensor(edge_weight, device=device)
    expert = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in pair]
    history = []
    last_mass_error = 0.
    for iteration in range(steps):
        for model, optimizer in zip(pair, optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
        logits = [[], []]
        for begin in range(0, len(paid), 512):
            block = {k: v[begin:begin+512] for k, v in data.items()}
            for i, model in enumerate(pair):
                logits[i].append(model(**block).gather(1, roles[begin:begin+512, None]).flatten())
        losses = [nn.functional.binary_cross_entropy_with_logits(torch.cat(z), target, reduction='none') for z in logits]
        reweighted = [peer_weights(loss-entropy, groups) for loss in losses]
        supervised = sum(.5*(losses[i]*reweighted[1-i]).mean() for i in range(2))
        offset = (iteration*128) % len(edge_weight)
        ids = (torch.arange(128, device=device)+offset) % len(edge_weight)
        before = [model(**{k: v[ids] for k, v in left.items()}).sigmoid() for model in pair]
        after = [model(**{k: v[ids] for k, v in right.items()}).sigmoid() for model in pair]
        regularizer = 0
        for i in range(2):
            delta = (before[i]-after[1-i].detach())**2 + (after[i]-before[1-i].detach())**2
            regularizer = regularizer+.5*(edge_weight[ids]*expert[1-i]*delta).mean()
        objective = supervised+regularizer
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite anchored reciprocal objective')
        objective.backward()
        for optimizer in optimizers:
            optimizer.step()
        if iteration == 0 or (iteration+1) % 1000 == 0 or iteration+1 == steps:
            counts = torch.bincount(groups).to(target.dtype)
            last_mass_error = max(float((torch.zeros_like(counts).index_add_(0, groups, w)-counts).abs().max())
                                  for w in reweighted)
            assert last_mass_error < .001
            history.append(dict(steps=iteration+1, objective=float(objective.detach()),
                                teacher_component=float(supervised.detach()),
                                visual_component=float(regularizer.detach()),
                                query_role_supervision_mass_maximum_error=last_mass_error))
    for model in pair:
        model.eval()
    risk = teacher_risks(pair, inputs, paid, device, hard=False)
    return dict(steps=steps, teacher_rows=len(paid), teacher_risks=risk,
                maximum_excess_KL=max(v['excess_KL'] for row in risk for v in row.values()),
                primary_teacher_interpolation_gate_not_imposed=True,
                noisy_teacher_denoising_is_a_hypothesis_not_proven_correctness=True,
                ordinary_parent_teacher_fidelity_gates_unchanged=True,
                peer_weight_floor=.2, original_query_role_mass_preserved=True,
                optimization_history=history, wall_seconds=time.perf_counter()-start,
                seed=0, no_manual_GT=True, peer_opinions_not_ground_truth=True)
