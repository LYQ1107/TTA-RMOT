"""Unlabelled visual coverage and teacher-anchored harmonic inference.

The graph is made solely from frozen image/query features. Teacher probabilities
are observations, not ground truth. No evaluation code is imported here.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.spatial.distance import cdist


def visual_graph(features, neighbours=3):
    x = np.asarray(features, dtype=float)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    distance = np.maximum(cdist(x, x, metric='sqeuclidean'), 0.)
    n = len(x)
    if n < 2:
        return np.zeros((n, n)), distance
    k = min(neighbours, n-1)
    ranked = np.argsort(distance + np.eye(n)*1e6, axis=1, kind='stable')[:, :k]
    scale = np.maximum(np.take_along_axis(distance, ranked[:, -1:], axis=1).ravel(), 1e-6)
    weights = np.exp(-distance / np.maximum(np.sqrt(scale[:, None]*scale[None]), 1e-6))
    connected = np.zeros((n, n), dtype=bool)
    np.put_along_axis(connected, ranked, True, axis=1)
    weights *= connected | connected.T
    np.fill_diagonal(weights, 0.)
    # Symmetric degree normalization keeps the graph penalty independent of
    # the number of duplicate candidate tracks; it does not use identity GT.
    degree = np.maximum(weights.sum(1), 1e-8)
    weights /= np.sqrt(degree[:, None]*degree[None])
    return weights, distance


def acquisition_order(prior, distance, eligible, method, rng):
    """Return distinct tracks: query-balanced budgets are applied by caller."""
    eligible = list(map(int, eligible))
    tie = rng.random(len(prior))
    ranked = sorted(eligible, key=lambda t: (-prior[t], tie[t], t))
    if method == 'PRIOR':
        return ranked
    if method == 'UNIFORM':
        return [eligible[i] for i in rng.permutation(len(eligible))]
    if method == 'UNCERTAINTY':
        return sorted(eligible, key=lambda t: (abs(prior[t]-.5), tie[t], t))
    if method != 'COVER':
        raise ValueError(method)
    strata = [list(a) for a in np.array_split(ranked, 3)]
    selected = []
    while len(selected) < len(eligible):
        for candidates in strata:
            remaining = [int(t) for t in candidates if t not in selected]
            if not remaining:
                continue
            if not selected:
                chosen = min(remaining, key=lambda t: (-prior[t], tie[t], t))
            else:
                chosen = min(remaining, key=lambda t: (-min(distance[t, selected]), tie[t], t))
            selected.append(chosen)
    return selected


def infer(prior, features, teacher_probability, *, correction=True, graph=True):
    """Minimize a quadratic graph energy with weak source and teacher anchors.

    No learned reliability or calibrated probability claim is made for these
    fixed precision constants. They are registered before target evaluation.
    """
    prior = np.asarray(prior, dtype=float)
    answer = np.asarray(teacher_probability, dtype=float)
    seen = np.isfinite(answer)
    source = np.clip(logit(np.clip(prior, .05, .95)), -2., 2.)
    evidence = np.zeros_like(prior)
    evidence[seen] = np.clip(logit(np.clip(answer[seen], .02, .98)), -4., 4.)
    # One teacher judgement may overturn the source. Repeated/correlated views
    # are deliberately not multiplied as independent observations.
    precision = np.zeros_like(prior)
    precision[seen] = 1. + 3.*np.abs(2.*answer[seen]-1.)
    source_precision = np.full_like(prior, .1)
    if correction:
        source_precision[seen] /= 1.+precision[seen]
    else:
        # Keep the exact same graph and teacher precision. Only the anchor
        # update changes, so this does not also ablate graph propagation.
        evidence[seen] = source[seen]+np.clip(evidence[seen],-2.,2.)
    weights, _ = visual_graph(features)
    if not graph:
        weights[:] = 0.
    laplacian = np.diag(weights.sum(1))-weights
    system = laplacian + np.diag(source_precision+precision)
    target = source_precision*source + precision*evidence
    solution = np.linalg.solve(system, target)
    return expit(solution)


def refine_distilled_prior(prior, features, teacher_probability, *, graph=True,source_weight=1.,node_mask=None,teacher_weight_factor=None,weights_override=None):
    """V2: target-teacher-trained unary, known-answer anchors, classic graph.

    A queried node has only its teacher anchor, rather than counting its
    fitted classifier prediction as an independent second observation.
    """
    prior=np.asarray(prior,dtype=float);answer=np.asarray(teacher_probability,dtype=float)
    seen=np.isfinite(answer)
    unary=np.clip(logit(np.clip(prior,1e-6,1-1e-6)),-4.,4.)
    source_precision=np.full_like(prior,source_weight);source_precision[seen]=0.
    precision=np.zeros_like(prior);precision[seen]=1.+3.*np.abs(2.*answer[seen]-1.)
    if teacher_weight_factor is not None:precision*=np.asarray(teacher_weight_factor)
    target=np.zeros_like(prior);target[seen]=np.clip(logit(np.clip(answer[seen],.02,.98)),-4.,4.)
    if weights_override is not None:weights=np.asarray(weights_override,dtype=float).copy()
    elif node_mask is None:weights,_=visual_graph(features)
    else:
        node_mask=np.asarray(node_mask,dtype=bool);weights=np.zeros((len(prior),len(prior)))
        selected=np.flatnonzero(node_mask);local,_=visual_graph(np.asarray(features)[selected])
        weights[np.ix_(selected,selected)]=local
    if not graph:weights[:]=0.
    laplacian=np.diag(weights.sum(1))-weights
    value=np.linalg.solve(laplacian+np.diag(source_precision+precision),source_precision*unary+precision*target)
    return expit(value)
