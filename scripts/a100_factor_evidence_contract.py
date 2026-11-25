"""Material input/representation controls for the new primary pilot; no GT."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from flex_uda.active.factor_evidence import FactorStudent, OrdinaryFactorKD, MODES, PROMPT_SHA
    torch.set_num_threads(2); torch.manual_seed(0)
    model = FactorStudent().eval(); ordinary = OrdinaryFactorKD().eval()
    bags = torch.randn(9, 3, 1101); current = torch.randn(9, 1101)
    present = torch.ones(9, 3, dtype=torch.bool); present[0, 1:] = False
    coarse = torch.randn(9, 1101)
    with torch.no_grad():
        base = model(bags, present, current, coarse)
        changed_bags = bags.clone(); changed_current = current.clone()
        changed_bags[..., -18:] += 9; changed_current[..., -18:] -= 13
        changed = model(changed_bags, present, changed_current, coarse)
        assert torch.equal(base[:, 1], changed[:, 1]), 'Appearance used explicit geometry/time'
        assert not torch.equal(base[:, 2], changed[:, 2]), 'State did not see current state'
        order = torch.tensor([2, 0, 1])
        reordered = model(bags[:, order], present[:, order], current, coarse)
        permutation_error = float((base-reordered).abs().max())
        assert permutation_error < 1e-6, permutation_error
        corrupted = bags.clone(); corrupted[0, 1:] = 1000
        assert torch.equal(base[0], model(corrupted, present, current, coarse)[0]), 'Padded anchors leaked'
        assert base.shape == ordinary(bags, present, current, coarse).shape == (9, len(MODES))
        full_input_effect = float((ordinary(bags, present, current, coarse)
                                  - ordinary(changed_bags, present, changed_current, coarse)).abs().max())
        assert full_input_effect > 0, 'Ordinary model was denied state inputs'
    # The ordinary comparator has access to each independent teacher role.
    # This target deliberately cannot be represented by multiplying its two
    # factor probabilities; the architecture does not impose that assumption.
    output = ordinary(bags, present, current, coarse)
    target = torch.tensor([.9, .1, .1]).expand_as(output)
    optimizer = torch.optim.Adam(ordinary.parameters(), lr=.003)
    for _ in range(400):
        optimizer.zero_grad(); z = ordinary(bags, present, current, coarse)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        probability = ordinary(bags, present, current, coarse).sigmoid()
        error = float((probability-target).abs().max()); assert error < .01, error
    print(json.dumps(dict(status='PASS_NEW_FACTOR_PRIMARY_MATERIAL_CONTROLS',
        explicit_geometry_invariance_of_appearance=True, padded_memory_ignored=True,
        memory_permutation_error=permutation_error, ordinary_has_full_information=True,
        three_independent_teacher_roles_fit_error=error, prompt_sha256=PROMPT_SHA,
        synthetic_only=True, target_GT_read=False, no_quality_claim=True)), flush=True)


if __name__ == '__main__': main()
