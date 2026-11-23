"""State exactly which replication invariance the exposure risk has.

Uniformly replicating a whole bag preserves its weighted empirical measure.
Replicating one member of a heterogeneous bag generally does not. Actual
identical request IDs are removed before either learning risk is built.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.query_profile import exposure_balanced_weights,supported_query_profile_kl
from scripts.a100_six_direction import PUBLIC


def main():
    p=torch.tensor([.2,.8,.3],dtype=torch.float64)
    y=torch.tensor([.9,.4,.7],dtype=torch.float64)
    bag=torch.tensor([0,0,1]);confidence=1+3*(2*y-1).abs()
    def evaluate(indices):
        b=bag[indices];w=exposure_balanced_weights(confidence[indices],b)
        loss=supported_query_profile_kl(p[indices].log(),y[indices],y[indices],torch.zeros_like(b),w)
        mass=torch.zeros(2,dtype=torch.float64).scatter_add(0,b,w)
        return float(loss),mass.tolist()
    original,original_mass=evaluate(torch.tensor([0,1,2]))
    whole,whole_mass=evaluate(torch.tensor([0,1,0,1,2]))
    partial,partial_mass=evaluate(torch.tensor([0,0,1,2]))
    assert abs(original-whole)<1e-12
    assert torch.allclose(torch.tensor(original_mass),torch.tensor(whole_mass))
    assert abs(original-partial)>1e-6 and original_mass!=partial_mass
    report=dict(status='PASS_QUERY_PROFILE_REPLICATION_SCOPE_AND_COUNTEREXAMPLE',
        scope='Uniformly replicate every observation in a bag by the same factor; keep targets, predictions, confidence and eligibility fixed.',
        original_loss=original,whole_bag_replicated_loss=whole,whole_bag_KL_error=abs(original-whole),
        original_bag_weight_mass=original_mass,whole_bag_replicated_weight_mass=whole_mass,
        partial_heterogeneous_duplicate_loss=partial,partial_duplicate_weight_mass=partial_mass,
        arbitrary_member_replication_invariance_explicitly_false=True,
        exact_request_repeat_deduplication_is_separate_preprocessing=True,
        one_over_count_is_not_inverse_propensity_or_unbiased_population_risk=True,
        earlier_audit_scope='The first check duplicated the only observation in one bag. It did not test arbitrary partial replication within a heterogeneous bag.',
        no_GT_or_teacher_or_images_or_trained_head_read=True,new_GPU_or_training_calls=0,
        source_sha256={name:file_sha(ROOT/name) for name in ('scripts/a100_query_profile_duplication_scope.py','flex_uda/active/query_profile.py','flex_uda/active/conditional_observation.py')},
        no_live_numerical_source_changed=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/'query_profile_duplication_scope.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
