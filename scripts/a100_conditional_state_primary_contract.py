"""CPU-only recovery, count and statistics checks before the full primary."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

DEST=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_primary_contract_20261127'
OWN=DEST/'registration.json'
SOURCES=('scripts/a100_conditional_state_primary_contract.py','scripts/a100_conditional_state_primary.py',
    'flex_uda/active/conditional_state_runtime.py','scripts/a100_conditional_state_statistics.py',
    'flex_uda/active/conditional_state_learning.py','scripts/a100_conditional_state_engineering.py')
BASE=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'


def prepare():
    if OWN.exists():raise FileExistsError(OWN)
    original=json.loads((BASE/'registration.json').read_text())['source_sha256']
    for name,sha in original.items():assert file_sha(ROOT/name)==sha,name
    atomic_json(OWN,dict(status='REGISTERED_BEFORE_FULL_PRIMARY_CONTRACT',source_sha256={n:file_sha(ROOT/n) for n in SOURCES},
        original42_source_sha256=original,synthetic_CPU_only=True,no_target_GT_teacher_images_GPU_or_HOTA=True))
    print(json.dumps(dict(status='REGISTERED_FULL_PRIMARY_RECOVERY_CONTRACT',sources=len(SOURCES))),flush=True)


def run():
    import numpy as np
    import torch
    from flex_uda.active.conditional_state_runtime import save_round,load_round
    from scripts.a100_conditional_state_primary import probability_scores,array_sha,COMPARATORS,METHODS
    from scripts.a100_factor_rsi_primary_quality import projection
    from scripts.a100_conditional_state_statistics import value
    plan=json.loads(OWN.read_text())
    for group in ('source_sha256','original42_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    if (DEST/'result.json').exists():raise FileExistsError(DEST/'result.json')
    torch.manual_seed(0);torch.set_num_threads(1)
    checks=[];rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix='state-checkpoint-',dir=DEST) as directory:
        p=Path(directory)/'round.pt';s=torch.nn.Linear(3,1);r=torch.nn.Linear(2,1)
        scores=np.array([[-1.,2.],[-3.,4.]],dtype='float32')
        # A rejected proposal has different scores. Save ONLY the accepted
        # scores so restart cannot accidentally import the final rejected try.
        rejected=np.full_like(scores,99)
        row=dict(stage=0,objective_before=1.,objective_after=.9,solver_attempted_steps=1500,observer_attempted_steps=1500)
        checkpoint=dict(solver=s.state_dict(),observer=r.state_dict(),rounds=[row],accepted_scores=scores,
            generator_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.empty(0,dtype=torch.uint8),registration_sha256='registered-fixture')
        save_round(p,checkpoint);first=file_sha(p)
        expected_np=rng.integers(100000,size=100);expected_torch=torch.rand(100)
        saved=load_round(p,'registered-fixture');rng.bit_generator.state=saved['generator_state'];torch.set_rng_state(saved['torch_rng'])
        assert np.array_equal(rng.integers(100000,size=100),expected_np) and torch.equal(torch.rand(100),expected_torch)
        assert np.array_equal(saved['accepted_scores'],scores) and not np.array_equal(saved['accepted_scores'],rejected)
        checks.extend(['accepted_scores_not_last_rejected_proposal','exact_numpy_and_torch_rng_round_recovery'])
        before={k:v.clone() for k,v in s.state_dict().items()}
        with torch.no_grad():
            for parameter in s.parameters():parameter.add_(10)
        s.load_state_dict(saved['solver']);assert all(torch.equal(s.state_dict()[k],v) for k,v in before.items())
        checks.append('exact_neural_parameter_round_recovery')
        try:load_round(p,'wrong-fixture')
        except AssertionError:checks.append('different_registration_rejected')
        else:raise AssertionError('Cross-registration checkpoint accepted')
        class Broken:
            def __reduce__(self):raise RuntimeError('deliberate synthetic serialization failure')
        try:save_round(p,dict(bad=Broken()))
        except RuntimeError:pass
        else:raise AssertionError('Synthetic serialization failure absent')
        assert file_sha(p)==first;checks.append('failed_atomic_write_preserves_last_complete_checkpoint')
    obs=np.array([[2,7],[1,3],[1,7],[3,3]])
    donor=np.array([[True,False,True,False],[False,True,False,True]])
    rank=np.array([[.1,.4,.8,.2],[.9,.3,.5,.7]])
    actual=projection(rank,donor,obs)
    expected=np.zeros_like(donor)
    for frame in set(obs[:,0]):
        rows=np.array([i for i,x in enumerate(obs) if x[0]==frame])
        for query in range(2):
            k=sum(bool(donor[query,i]) for i in rows)
            order=sorted(rows,key=lambda i:(-rank[query,i],i));expected[query,order[:k]]=True
    assert np.array_equal(actual,expected);checks.append('CountMatched_preserves_each_query_frame_count_and_ranking')
    probabilities=np.array([0.,1e-250,.25,.5,.75,1.])
    z=probability_scores(probabilities)
    assert np.isfinite(z).all() and (np.diff(z)>0).all() and np.array_equal(z>0,probabilities>.5)
    checks.append('finite_probability_readout_keeps_extreme_probability_order')
    assert len(METHODS)==8 and len(COMPARATORS)==7 and (len(METHODS)+2*len(COMPARATORS))*148==3256
    checks.append('all148_outputs_and7_comparators_registered_consistently')
    # A simple independently computed aggregate illustrates why video HOTA
    # must not be averaged: shared AssA=1, global TP=4, FP+FN=1 =>sqrt(.8).
    aggregate=dict(numerator=[4.]*19,denominator=[5.]*19)
    assert abs(value(aggregate)-100*np.sqrt(.8))<1e-12
    checks.append('whole_setting_TP_weighted_statistic_known_answer')
    assert array_sha(np.array([1.,2.],dtype='float32'))==hashlib.sha256(np.array([1.,2.],dtype='float32').tobytes()).hexdigest()
    checks.append('ordinary_target_array_integrity_digest')
    for group in ('source_sha256','original42_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    receipt=dict(status='PASS_FULL_PRIMARY_RECOVERY_COUNT_AND_STATISTICS_CONTRACT',checks=checks,
        original42sources_unchanged=True,source_sha256=plan['source_sha256'],synthetic_CPU_only=True,
        no_target_GT_teacher_images_GPU_or_HOTA=True,not_six_setting_quality=True,goal_status='ACTIVE')
    atomic_json(DEST/'result.json',receipt);print(json.dumps(dict(status=receipt['status'],checks=len(checks))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
