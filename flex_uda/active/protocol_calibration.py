"""Task-contract pooling and a matched, conventional nonempty baseline.

The contract is provided by the task, never inferred from target GT. Pooling
over all scene candidates is enabled only for CHECKLIST. PRESUPPOSED retains
the ordinary unary exactly. Visual calibration remains a separate operation.
"""
import numpy as np
from scipy.special import expit, logit


def global_pool(unary, teacher, weights, valid, supported):
    """Solve the existing equal-edge graph energy by a rank-one identity.

    This is the same operator as topology_selection.apply_graph with its
    global graph, including raw known-answer anchors and .1 unknown precision.
    Only this solve is linear in node count; constructing the matched graph
    normalization can still be quadratic.
    """
    result = np.array(unary, copy=True)
    for q in np.flatnonzero(supported):
        p = np.asarray(unary[q], dtype=float)
        answer = np.asarray(teacher[q], dtype=float)
        known = np.isfinite(answer)
        source = np.full(p.shape, .1); source[known] = 0.
        precision = np.zeros(p.shape)
        precision[known] = 1. + 3.*np.abs(2.*answer[known]-1.)
        anchor = np.zeros(p.shape)
        anchor[known] = np.clip(logit(np.clip(answer[known], .02, .98)), -4., 4.)
        diagonal = source + precision
        target = source*np.clip(logit(np.clip(p, 1e-6, 1-1e-6)), -4., 4.) + precision*anchor
        value = target/diagonal
        ids = np.flatnonzero(valid[q]); n = len(ids)
        if n > 1:
            block = np.asarray(weights[q])[np.ix_(ids, ids)]
            alpha = float(block[0, 1])
            expected = alpha*(np.ones((n, n))-np.eye(n))
            assert np.array_equal(block, expected), 'global_pool requires equal off-diagonal edges'
            inverse = 1./(diagonal[ids]+alpha*n)
            local = target[ids]*inverse
            value[ids] = local + inverse*(alpha*local.sum()/(1.-alpha*inverse.sum()))
        result[q] = expit(value)
    return result


def contract_graph(unary, global_probability, protocol):
    if protocol == 'PRESUPPOSED':
        return np.array(unary, copy=True)
    if protocol == 'CHECKLIST':
        return np.array(global_probability, copy=True)
    raise ValueError('Unknown task contract: '+str(protocol))


def nonempty_baseline(unary, protocol):
    """Ordinary independent Bernoulli marginals conditioned on >=1 target.

    The supplied PRESUPPOSED contract concerns the scene. Applying it to the
    fixed candidate set assumes candidate coverage; it is a control rather
    than an oracle guarantee. No singleton constraint or target-tuned cutoff.
    """
    result = np.array(unary, copy=True)
    if protocol == 'CHECKLIST':
        return result
    if protocol != 'PRESUPPOSED':
        raise ValueError('Unknown task contract: '+str(protocol))
    p = np.clip(np.asarray(unary, dtype=float), 0., 1.)
    if not p.shape[1]:
        return result
    with np.errstate(divide='ignore'):
        exists = -np.expm1(np.log1p(-p).sum(axis=1))
    for q, denominator in enumerate(exists):
        result[q] = p[q]/denominator if denominator > 0 else 1./p.shape[1]
    return result
