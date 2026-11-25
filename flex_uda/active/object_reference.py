"""Candidate object/reference learning from a structured teacher state model.

Unlike perturbing an ordinary composite score, this transports the training
targets before fitting the reference head. The construction is conditional on
an additive logit state response, not a claim of semantic or HOTA correctness.
"""
import numpy as np
from scipy.special import expit, logit

REFERENCE_POSITION = .5


def oriented_position(coords, direction):
    x = np.asarray(coords, float)[..., 0]
    d = np.asarray(direction)
    return np.where(d > 0, x, np.where(d < 0, 1.-x, REFERENCE_POSITION))


def transport_targets(probability, coords, direction, beta):
    """Transport actual pseudo-targets, keeping inactive values exactly equal."""
    assert np.isfinite(beta) and beta >= 0
    raw = np.clip(np.asarray(probability, float), .02, .98)
    d = np.asarray(direction)
    assert raw.shape == d.shape
    assert np.asarray(coords).shape == raw.shape + (4,)
    assert np.isfinite(raw).all() and np.isfinite(coords).all()
    assert np.isin(d, (-1, 0, 1)).all()
    result = raw.copy()
    active = (d != 0) & (beta > 0)
    position = oriented_position(coords, d)
    result[active] = expit(logit(raw[active]) - beta*(position[active]-REFERENCE_POSITION))
    return result


def grouped_targets(cells, raw_probability, coords, directions, beta):
    """Retain the exact weighted Bernoulli loss after grouping frame targets.

    A stable head has one logit per (query, provided TrackID); cross entropy is
    affine in its target. Summed original weights and weighted target means
    therefore aggregate that objective exactly. They do not claim independent
    frames, statistical noise removal, or equivalence of target entropies.
    """
    cells = np.asarray(cells, int)
    raw_probability = np.asarray(raw_probability, float)
    assert cells.ndim == 2 and cells.shape[1] == 3 and len(cells) == len(raw_probability)
    directions = np.asarray(directions, int)
    row_direction = directions[cells[:, 0]]
    reference = transport_targets(raw_probability, coords, row_direction, beta)
    control = np.clip(raw_probability, .02, .98)
    weight = 1.+3.*np.abs(2.*raw_probability-1.)
    unique, inverse = np.unique(cells[:, :2], axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weight, minlength=len(unique))
    y = np.bincount(inverse, weights=weight*reference, minlength=len(unique))/mass
    unshifted = np.bincount(inverse, weights=weight*control, minlength=len(unique))/mass
    assert mass.min() > 0 and np.isfinite(y).all()
    return dict(cells=unique, reference_target=y, unshifted_target=unshifted,
        sample_weight=mass, frame_to_group=inverse, reference_frame_target=reference,
        raw_frame_target=control, raw_frame_weight=weight)


def reconstruct(reference_probability, position, beta, eligible=None):
    """Reconstruct from a learned reference, without a composite-score cap."""
    assert np.isfinite(beta) and beta >= 0
    reference = np.asarray(reference_probability)
    position = np.asarray(position, float)
    result = reference.astype(float, copy=True)
    if beta == 0:
        return reference.copy()
    active = np.ones(np.broadcast_shapes(reference.shape, position.shape), bool)
    if eligible is not None:
        active &= np.broadcast_to(np.asarray(eligible, bool), active.shape)
    shift = beta*(position-REFERENCE_POSITION)
    active &= np.broadcast_to(shift != 0, active.shape)
    assert active.shape == reference.shape
    assert np.isfinite(position).all() and ((position >= 0) & (position <= 1)).all()
    changed = expit(logit(np.clip(reference, 1e-7, 1.-1e-7)) + shift)
    result[active] = changed[active]
    return result
