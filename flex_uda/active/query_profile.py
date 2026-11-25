"""Supported query-level preference learning, without a one-target decoder.

Condition on an actually supported query/view stratum, then learn allocation
of positive evidence across queried candidates/observations. Scores remain
independent Bernoulli outputs at inference; this is not a softmax tracker.
"""
import torch
from flex_uda.active.conditional_observation import relative_choice_kl


def exposure_balanced_weights(confidence, bag_view_group):
    """Each queried bag/view gets its mean confidence, not repeat count."""
    counts = torch.bincount(bag_view_group)
    return confidence/counts[bag_view_group].to(confidence.dtype)


def supported_query_profile_kl(log_probability, clipped_target, raw_teacher,
                               query_view_group, balanced_weight):
    """Only assign preferences after at least one actual teacher Yes.

    The caller supplies whole query/view groups in a batch and weights
    computed from all unique observed states of each bag/view. A group of
    all No responses is unknown for this preference objective, not absent.
    Teacher Yes is guidance, not a ground-truth existence label.
    """
    n = int(query_view_group.max())+1
    counts = torch.bincount(query_view_group, minlength=n)
    maximum = torch.full((n,), -torch.inf, dtype=raw_teacher.dtype, device=raw_teacher.device)
    maximum = maximum.scatter_reduce(0, query_view_group, raw_teacher, reduce='amax', include_self=True)
    supported = (maximum >= .5) & (counts >= 2)
    rows = supported[query_view_group]
    if not bool(rows.any()):
        return log_probability.sum()*0
    _, group = torch.unique(query_view_group[rows], sorted=True, return_inverse=True)
    return relative_choice_kl(log_probability[rows], clipped_target[rows], group, balanced_weight[rows])
