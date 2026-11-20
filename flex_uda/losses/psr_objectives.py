"""Separate normalized source/direct/weak-expanded objectives; no count inflation."""
from .distillation import soft_distillation


def psr_losses(direct_margin,direct_probability,direct_valid,expanded_margin=None,expanded_probability=None,expanded_valid=None,expanded_weight=.25):
    direct=soft_distillation(direct_margin,direct_probability,direct_valid)
    expanded=direct*0 if expanded_margin is None else soft_distillation(expanded_margin,expanded_probability,expanded_valid)
    return dict(direct=direct,expanded=expanded,total=direct+expanded_weight*expanded)
