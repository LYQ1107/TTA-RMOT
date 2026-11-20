"""Three-valued conjunction and pre-registered reliability, never target labels."""
import math
from .teacher_types import EvidenceDecision,EvidenceState


def binary_entropy(p):
    p=min(1-1e-12,max(1e-12,float(p)))
    return -p*math.log(p)-(1-p)*math.log(1-p)


def confidence_decision(probability,valid,positive=0.9,negative=0.1):
    if not valid:return EvidenceDecision(EvidenceState.UNKNOWN,probability,['TEACHER_CANNOT_ANSWER'])
    if probability>=positive:return EvidenceDecision(EvidenceState.SUPPORT_POS,probability)
    if probability<=negative:return EvidenceDecision(EvidenceState.SUPPORT_NEG,probability)
    return EvidenceDecision(EvidenceState.UNKNOWN,probability,['LOW_RELIABILITY'])


def verify(attributes,observations,positive,negative,visual_min=0.03,spread_max=0.2):
    if not attributes:return EvidenceDecision(EvidenceState.UNKNOWN,0.5,['UNSUPPORTED_EXPRESSION'])
    states=[];features=[];reasons=[];probabilities=[]
    for attribute,views in zip(attributes,observations):
        original=views.get('original');previous=views.get('previous_observation')
        if original is None:
            states.append(EvidenceState.UNKNOWN);probabilities.append(0.5);reasons.append('ATTRIBUTE_NOT_VISIBLE');continue
        p=original['yes_probability'];probabilities.append(p)
        if not original['top1_is_yes_no']:
            states.append(EvidenceState.UNKNOWN);reasons.append('TEACHER_CANNOT_ANSWER');continue
        if any(views.get(k) is None for k in ('no_image','candidate_occluded','background_occluded','previous_observation')):
            states.append(EvidenceState.UNKNOWN);reasons.append('TEMPORAL_EVIDENCE_MISSING' if previous is None else 'ATTRIBUTE_NOT_VISIBLE');continue
        prior=views['no_image']['yes_probability'];body=views['candidate_occluded']['yes_probability']
        bg=views['background_occluded']['yes_probability'];past=previous['yes_probability']
        f=dict(attribute=attribute['text'],original=p,no_image=prior,candidate_occluded=body,
            background_occluded=bg,previous_observation=past,visual_delta=p-prior,
            target_probability_change=p-body,background_probability_change=p-bg,
            target_sensitivity_excess=abs(p-body)-abs(p-bg),entropy_delta=binary_entropy(prior)-binary_entropy(p),
            cross_observation_spread=abs(p-past))
        features.append(f)
        if not previous['top1_is_yes_no'] or abs(p-past)>spread_max:
            states.append(EvidenceState.UNKNOWN);reasons.append('TEACHER_DISAGREEMENT');continue
        if abs(p-bg)>spread_max or f['target_sensitivity_excess']<0:
            states.append(EvidenceState.UNKNOWN);reasons.append('CONTRADICTORY_VISUAL_EVIDENCE');continue
        if p>=positive and past>=positive and p-prior>=visual_min:
            states.append(EvidenceState.SUPPORT_POS)
        elif p<=negative and past<=negative and prior-p>=visual_min:
            states.append(EvidenceState.SUPPORT_NEG)
        else:states.append(EvidenceState.UNKNOWN);reasons.append('LOW_RELIABILITY')
    state=(EvidenceState.SUPPORT_NEG if EvidenceState.SUPPORT_NEG in states else
        EvidenceState.SUPPORT_POS if all(s==EvidenceState.SUPPORT_POS for s in states) else EvidenceState.UNKNOWN)
    return EvidenceDecision(state,min(probabilities),sorted(set(reasons)),dict(attributes=features,
        reliability_score=min((abs(f['visual_delta'])*max(0.,f['target_sensitivity_excess'])*(1-f['cross_observation_spread']) for f in features),default=0.),
        intervention_warning='occlusion may be OOD; sensitivity is a selection feature, not semantic truth',
        temporal_warning='adjacent overlapping histories are correlated; physical ID is not a timeless semantic label'))
