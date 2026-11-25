"""Appearance-conditioned evidence transfer on supplied observation nodes.

These classical graph operators never create boxes, change a Track ID, learn a
visual backbone, or open evaluation labels. A teacher answer anchors its actual
observation instead of asserting that an entire possibly mixed track is alike.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


def observation_edges(features, node_track, offsets, visible):
    x = np.asarray(features, dtype=float)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    distance = np.maximum(cdist(x, x, metric='sqeuclidean'), 0.)
    n = len(x)
    track = np.asarray(node_track)
    visible = np.asarray(visible, dtype=bool)
    allowed = (track[:, None] != track[None, :]) & visible[:, None] & visible[None, :]
    connected = np.zeros((n, n), dtype=bool)
    scale = np.full(n, 1e-6)
    for i in range(n):
        choices = np.flatnonzero(allowed[i])
        nearest = choices[np.argsort(distance[i, choices], kind='stable')[:3]]
        if len(nearest):
            connected[i, nearest] = True
            scale[i] = max(float(distance[i, nearest[-1]]), 1e-6)
    visual = np.exp(-distance / np.maximum(np.sqrt(scale[:, None]*scale[None, :]), 1e-6))
    visual *= connected | connected.T
    degree = np.maximum(visual.sum(1), 1e-8)
    visual /= np.sqrt(degree[:, None]*degree[None, :])
    temporal = np.zeros((n, n))
    pairs = [(i, i+1) for a, b in zip(offsets[:-1], offsets[1:])
             for i in range(int(a), int(b)-1) if visible[i] and visible[i+1]]
    for i, j in pairs:
        temporal[i, j] = temporal[j, i] = 1.
    degree = np.maximum(temporal.sum(1), 1.)
    temporal /= np.sqrt(degree[:, None]*degree[None, :])
    positive = [distance[i, j] for i, j in pairs if distance[i, j] > 1e-8]
    bandwidth = max(float(np.median(positive)) if positive else 1e-6, 1e-6)
    # Normalize the constant topology first. Normalizing a weighted two-node
    # edge would cancel its attenuation and silently erase the intended test.
    conditional = temporal * np.exp(-distance / bandwidth)
    matched = temporal * (conditional.sum()/temporal.sum() if temporal.sum() else 0.)
    weights = dict(ANCHOR_NO_GRAPH=np.zeros((n, n)),
        ANCHOR_VISUAL_ONLY=.5*visual,
        ANCHOR_TEMPORAL_ONLY=.5*conditional,
        ANCHOR_FULL=.5*(visual+conditional),
        CONSTANT_TEMPORAL=.5*(visual+temporal),
        MASS_MATCHED_CONSTANT_TEMPORAL=.5*(visual+matched))
    receipt = dict(nodes=n, visual_undirected_edges=int((visual>0).sum()//2),
        temporal_undirected_edges=len(pairs), temporal_bandwidth=bandwidth,
        constant_temporal_mass=float(temporal.sum()),
        conditional_temporal_mass=float(conditional.sum()),
        graph_only_uses_images=True)
    return weights, receipt


def anchored_solution(prior, teacher, weights):
    p = np.asarray(prior, dtype=float)
    answer = np.asarray(teacher, dtype=float)
    known = np.isfinite(answer)
    unary = np.clip(logit(np.clip(p, 1e-6, 1-1e-6)), -4., 4.)
    source_precision = np.full(len(p), .1)
    source_precision[known] = 0.
    teacher_precision = np.zeros(len(p))
    teacher_precision[known] = 1.+3.*np.abs(2.*answer[known]-1.)
    evidence = np.zeros(len(p))
    evidence[known] = np.clip(logit(np.clip(answer[known], .02, .98)), -4., 4.)
    weights = csr_matrix(weights)
    system = diags(np.asarray(weights.sum(1)).ravel()+source_precision+teacher_precision)-weights
    rhs = source_precision*unary + teacher_precision*evidence
    return expit(spsolve(system.tocsc(), rhs))


def interpolate_nodes(probabilities, node_frames, offsets, observations, tracks):
    probabilities = np.asarray(probabilities)
    result = np.empty((len(probabilities), len(observations)), dtype=float)
    values = logit(np.clip(probabilities, 1e-6, 1-1e-6))
    for ti, track in enumerate(tracks):
        rows = np.flatnonzero(observations[:, 1] == track)
        a, b = int(offsets[ti]), int(offsets[ti+1])
        frames = node_frames[a:b]
        for qi in range(len(values)):
            result[qi, rows] = expit(np.interp(observations[rows, 0], frames, values[qi, a:b]))
    return result
