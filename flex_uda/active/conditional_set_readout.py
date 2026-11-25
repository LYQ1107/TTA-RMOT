"""Conditional set decision from learned utilities and an external count.

For P(S | K=k, X) proportional to exp(sum(u_i for i in S)), the MAP set
is top-k by u. A common utility offset cancels. This is a proposed readout
model, NOT the MAP of an arbitrary pair-interaction network, and carries
no guarantee of RMOT gains. Counts never come from manual target labels.
"""
import numpy as np


def at_donor_counts(utilities, donor_bits, observations):
    utilities = np.asarray(utilities, dtype='float64')
    donor_bits = np.asarray(donor_bits)
    observations = np.asarray(observations)
    assert utilities.shape == donor_bits.shape and utilities.ndim == 2
    assert donor_bits.dtype == bool and observations.shape == (utilities.shape[1], 2)
    assert np.isfinite(utilities).all()
    result = np.zeros_like(donor_bits)
    order = np.argsort(observations[:, 0], kind='stable')
    boundaries = np.flatnonzero(np.diff(observations[order, 0]))+1
    for ids in np.split(order, boundaries):
        for qi in range(len(utilities)):
            k = int(donor_bits[qi, ids].sum())
            selected = ids[np.lexsort((observations[ids, 1], -utilities[qi, ids]))[:k]]
            result[qi, selected] = True
        assert np.array_equal(result[:, ids].sum(1), donor_bits[:, ids].sum(1))
    return result
