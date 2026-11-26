"""Material online-RSI controls: paid scope, reward and finite policies, no GT."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
    from flex_uda.active.factor_evidence import FactorStudent
    from flex_uda.active.factor_rsi_learning import (Observer, Inputs, make_pool, probes_and_seeds,
        choose, uid, discrimination_credit, teacher_floor, trust_update)
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from scripts.a100_observation_design_prepare import OUT as PREP
    from scripts.a100_factor_rsi_primary_pilot import OUT
    from flex_tta.runtime.pf_receipts import atomic_json
    torch.set_num_threads(2); torch.manual_seed(0); guard = generation_guard()
    use_verified_local_features(); rows = []
    for pair in ('C1', 'C2', 'C4'):
        target = PAIRS[pair][1]; video = sorted(metadata(target)['videos'])[0]
        original = json.loads((PREP/pair/video/'UNIFORM_requests.json').read_text())['requests']
        quotas = dict(Counter(int(r['local_query']) for r in original))
        v = JointObservationVideo(pair, video); inputs = Inputs(v); pool = make_pool(v, quotas)
        probes, seeds, remaining, reserved = probes_and_seeds(v, original, quotas, pool)
        assert len({uid(r) for r in seeds}) == len(seeds)
        assert not ({uid(r) for r in seeds} & {uid(r) for p in probes for r in p})
        assert len(seeds)+6+sum(remaining.values()) == len(original)
        assert sum(len({uid(r) for r in p}) for p in probes) == 6
        actor = Observer(); solver = FactorStudent().eval()
        # Test a real finite query decision, not thousands of optimizer steps.
        q = next(q for q,n in remaining.items() if n)
        selected, trace = choose(actor, solver, inputs, pool, {q:2}, seeds, reserved, np.random.default_rng(0))
        assert len({uid(r) for r in selected}) == 2
        assert all((r['local_query'],r['track_id'],r['frame_id']) not in reserved for r in selected)
        _, decisions = actor._distribution_trace
        for _, _, logp in decisions:
            assert abs(float(logp.exp().sum())-1) < 1e-12 and bool((logp.exp()>0).all())
        frozen = copy.deepcopy(actor); before = torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
        zero = trust_update(actor,None,None,1.,1.,0.,reward_override=0.)
        assert zero['actor_parameter_L2_change'] == 0
        assert torch.equal(before,torch.cat([p.detach().flatten() for p in actor.parameters()]))
        update = trust_update(actor,None,None,1.,.8,0.,reward_override=.15)
        assert update['accepted'] and 0 < update['mean_recorded_state_KL'] <= .01+1e-10
        assert update['actor_parameter_L2_change'] > 0
        frozen_update = trust_update(frozen,None,None,1.,.8,0.,reward_override=.15,frozen=True)
        assert frozen_update['actor_parameter_L2_change'] == 0
        rows.append(dict(pair=pair,video=video,budget=len(original),queries=len(quotas),pool=len(pool),
            probe_labels_excluded_from_current_fit=True,probability_full_support=True,update=update))
    answers = [dict(yes_probability=.8),dict(yes_probability=.2)]
    base = discrimination_credit(np.array([0.,0.]),np.array([1.,-1.]),answers)
    shifted = discrimination_credit(np.array([19.,19.]),np.array([24.,22.]),answers)
    assert base['reward'] > 0 and abs(base['reward']-shifted['reward']) < 1e-14
    equal = discrimination_credit(np.array([0.,0.]),np.array([1.,-1.]),[dict(yes_probability=.1)]*2)
    assert abs(equal['reward']) < 1e-14
    # Two paid views of exactly the same common input can conflict. Their
    # entropy of the mean, rather than mean individual entropy, is the floor.
    a=dict(bags=np.zeros((2,3,2)),present=np.ones((2,3)),current=np.zeros((2,2)),coarse=np.zeros((2,2)))
    floor=teacher_floor(a,np.array([0,0]),np.array([.1,.9]))
    assert abs(floor[0]-np.log(2)) < 1e-12 and np.isnan(floor[1:]).all()
    assert not guard
    receipt=dict(status='PASS_NEW_FACTOR_RSI_ONLINE_CONTRACT',cases=rows,zero_credit_no_actor_noise=True,
        frozen_actor_unchanged=True,common_logit_shift_reward_error=abs(base['reward']-shifted['reward']),
        exact_input_conflict_floor=float(floor[0]),seed=0,target_GT_read=False,
        synthetic_credit_not_RMOT_quality=True)
    atomic_json(OUT/'online_contract.json',receipt);print(json.dumps(receipt),flush=True)


if __name__ == '__main__': main()
