"""Reproduce a conditional counterexample to additive whole-query logits.

This synthetic algebra audit is neither target GT nor an RMOT quality study.
The probability chain rule is classical; a usable observation/learning
mechanism, not renaming the rule, would need to supply the contribution.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'docs/a100_six_direction_20261122'


def logit(p):
    return math.log(p)-math.log1p(-p)


def main():
    no_states = [(a, r) for a in (0, 1) for r in (0, 1) if not (a and r)]
    assert len(no_states) == 3 and (0, 1) in no_states and (1, 0) in no_states
    beta = 4.; z = .5; relation = 1./(1.+math.exp(-beta*(z-.5)))
    derivatives = []
    for attribute in (.2, .8):
        analytical = beta*(1.-relation)/(1.-attribute*relation)
        eps = 1e-5
        def whole_logit(position):
            r = 1./(1.+math.exp(-beta*(position-.5)))
            return logit(attribute*r)
        numerical = (whole_logit(z+eps)-whole_logit(z-eps))/(2*eps)
        assert abs(analytical-numerical) < 1e-8
        derivatives.append(dict(attribute_probability=attribute,
            whole_query_logit_state_derivative=analytical, numerical_derivative=numerical))
    assert abs(derivatives[0]['whole_query_logit_state_derivative']-
        derivatives[1]['whole_query_logit_state_derivative']) > 1.
    attribute, relation_if_attribute, relation_if_not = .4, .8, .1
    joint = attribute*relation_if_attribute
    marginal = attribute*relation_if_attribute+(1.-attribute)*relation_if_not
    independence_product = attribute*marginal
    assert abs(joint-independence_product) > .1
    report = dict(status='PASS_SYNTHETIC_CONJUNCTION_AND_ADDITIVE_LOGIT_COUNTEREXAMPLE',
        No_conjunction_compatible_states=no_states,
        chain_rule='P(A and R | I)=P(A | I)P(R | A,I); for positiveP(A|I); zero case has zero jointprobability',
        independence_is_not_assumed=dict(P_A=attribute, P_R_if_A=relation_if_attribute,
            P_R_if_not_A=relation_if_not, correct_joint=joint,
            incorrect_unconditional_product=independence_product),
        conditional_counterexample='IfP(A|I)=a is stable andP(R|A,z)=sigmoid(beta(z-c)), dlogitP(AandR|I,z)/dz=beta(1-r)/(1-ar).The slope generally depends on a andz; a shared additive whole-query logit slope is not implied by conjunction.',
        derivative_witness=derivatives,
        scope='Syntheticconditionalalgebra;notproofthatrealteacherfollowsproductlogisticmodel,notanimplementednewfactorlearner,notHOTAorsemanticaccuracy.',
        classical_probability_identity_is_not_claimed_novel=True,
        manual_target_GT_or_teacher_answers_read=0, new_GPU_or_quality_predictions=0,
        old_methods_or_outputs_changed=False, goal_status='ACTIVE')
    (PUBLIC/'compositional_reference_theory_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
