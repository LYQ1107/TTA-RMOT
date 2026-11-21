"""Image-only relative position and query-conditioned classical cohort graphs.

This module never reads teacher answers or target labels while constructing
features/edges. Supplied boxes and Track IDs remain unchanged.
"""
import re
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from flex_uda.active.acquisition import COLORS, expression_types
from flex_uda.active.color_cohort import part_mask,augmented_features


def context_part_mask(text):
    original = part_mask(text)
    if original.any():
        return original
    types = set(expression_types(text))
    if 'MOTION' in types or not ({'SPATIAL', 'ROLE'} & types):
        return original
    text = text.lower()
    if re.search(r'\b('+COLORS+r'|dark|light)\b', text):
        upper = bool(re.search(r'\b(tshirt|t-shirt|shirt|jersey|top|sweater)\b', text))
        lower = bool(re.search(r'\b(pants|trousers|skirt|shorts)\b', text))
        return np.asarray([0., float(upper), float(lower)]) if upper or lower else np.asarray([1., 0., 0.])
    return np.asarray([0., 1., 0.]) if re.search(r'\b(team|libero)\b', text) else np.asarray([1., 0., 0.])


def distillation_features(pcd,colours,texts,scope):
    """Exact natural768D or matched1080D teacher-head inputs."""
    if scope=='pcd':return np.asarray(pcd,dtype='float32')
    if scope=='zero':
        parsed=np.zeros(np.asarray(pcd).shape[:2]+(312,),dtype=np.asarray(pcd).dtype)
    elif scope=='clothing':
        parsed=augmented_features(pcd,colours,texts)[2]
    elif scope=='extended':
        parsed=np.stack([(colours*context_part_mask(text)[None,:,None]).reshape(len(colours),-1) for text in texts])
        parsed/=np.maximum(np.linalg.norm(parsed,axis=2,keepdims=True),1e-8)
    else:raise ValueError(scope)
    return np.concatenate([pcd,parsed],axis=2).astype('float32')


def relative_positions(bank):
    """Eight within-scene statistics; camera translation/scale cancel in ranks."""
    observations = np.asarray(bank['observations'])
    boxes = np.asarray([list(map(float, line.split(',')[2:6])) for line in bank['lines']])
    centres = boxes[:, :2]+.5*boxes[:, 2:4]
    values = np.zeros((len(boxes), 3))
    for frame in np.unique(observations[:, 0]):
        ids = np.flatnonzero(observations[:, 0] == frame)
        for axis in range(2):
            values[ids, axis] = (rankdata(centres[ids, axis], method='average')-.5)/len(ids)
        heights = np.log(np.maximum(boxes[ids, 3], 1))
        values[ids, 2] = heights-np.median(heights)
    output = []
    for track in np.unique(observations[:, 1]):
        ids = np.flatnonzero(observations[:, 1] == track)
        samples = ids[np.unique(np.linspace(0, len(ids)-1, min(3, len(ids))).round().astype(int))]
        anchor = samples[np.argmax(np.prod(boxes[samples, 2:4], axis=1))]
        local = values[ids]
        output.append(np.r_[local.mean(0), local.std(0), values[anchor, :2]])
    return np.asarray(output, dtype='float32')


def context_arrays(pcd, colours, texts, positions):
    pcd = np.asarray(pcd)
    centred = pcd-pcd.mean(2, keepdims=True)
    centred /= np.maximum(np.linalg.norm(centred, axis=2, keepdims=True), 1e-8)
    masks = np.stack([context_part_mask(text) for text in texts])
    parsed = np.stack([(colours*mask[None, :, None]).reshape(len(colours), -1) for mask in masks])
    parsed /= np.maximum(np.linalg.norm(parsed, axis=2, keepdims=True), 1e-8)
    spatial = np.asarray(['SPATIAL' in expression_types(text) for text in texts])
    raw = np.concatenate([pcd, parsed, positions[None, :, :]*spatial[:, None, None]], axis=2).astype('float32')
    geometry = np.concatenate([centred, parsed], axis=2)
    return raw, geometry, masks, spatial


def graph_from_distance(distance):
    n = len(distance)
    if n < 2:
        return np.zeros((n, n))
    k = min(3, n-1)
    nearest = np.argsort(distance+np.eye(n)*1e6, axis=1, kind='stable')[:, :k]
    scale = np.maximum(np.take_along_axis(distance, nearest[:, -1:], axis=1).ravel(), 1e-6)
    connected = np.zeros((n, n), dtype=bool)
    np.put_along_axis(connected, nearest, True, axis=1)
    weights = np.exp(-distance/np.maximum(np.sqrt(scale[:, None]*scale[None, :]), 1e-6))
    weights *= connected | connected.T
    np.fill_diagonal(weights, 0.)
    degree = np.maximum(weights.sum(1), 1e-8)
    return weights/np.sqrt(degree[:, None]*degree[None, :])


def context_weights(geometry, positions, visible, *, spatial):
    ids = np.flatnonzero(visible)
    n = len(geometry)
    feature = np.asarray(geometry)[ids].astype(float)
    feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-8)
    distance = np.maximum(cdist(feature, feature, 'sqeuclidean'), 0.)
    visual = graph_from_distance(distance)
    if spatial and len(ids) > 1:
        position = np.asarray(positions, dtype=float)[ids]
        position = (position-position.mean(0))/np.maximum(position.std(0), .05)
        other = np.maximum(cdist(position, position, 'sqeuclidean'), 0.)
        def scale(d):
            positive = d[d > 1e-12]
            return d/max(float(np.median(positive)) if len(positive) else 1., 1e-8)
        full = graph_from_distance(.5*scale(distance)+.5*scale(other))
    else:
        full = visual.copy()
    complete = np.ones_like(full)-np.eye(len(ids))
    if complete.sum():
        complete *= full.sum()/complete.sum()
    result = {}
    for name, local in [('VISUAL', visual), ('FULL', full), ('COMPLETE', complete)]:
        result[name] = np.zeros((n, n))
        result[name][np.ix_(ids, ids)] = local
    return result
