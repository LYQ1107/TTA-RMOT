"""Material reciprocal-gradient, memory, teacher-fold and risk controls."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.reciprocal_memory import (models,folds,risk_floor,update_experts,train,MemoryInputs)
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from scripts.a100_factor_rsi_primary_pilot import OUT as PARENT,verify_plan
    from scripts.a100_reciprocal_memory_pilot import OUT
    from flex_tta.runtime.pf_receipts import atomic_json
    guard=generation_guard();torch.set_num_threads(2);torch.manual_seed(0);generator=np.random.default_rng(0)
    a=dict(bags=generator.normal(size=(9,3,1101)).astype('float32'),present=np.ones((9,3),bool),
        current=generator.normal(size=(9,1101)).astype('float32'),coarse=generator.normal(size=(9,1101)).astype('float32'),
        memory=generator.normal(size=(9,2202)).astype('float32'))
    paid=[dict(row=i,factor_mode=('WHOLE','APPEARANCE','STATE')[i%3],yes_probability=.2+.6*(i%2)) for i in range(9)]
    class Toy:
        def arrays(self,records):return {k:v[[r['row'] for r in records]] for k,v in a.items()}
    ordinary=models('cpu');mutual=models('cpu')
    for first,second in zip(ordinary,mutual):
        assert all(torch.equal(first.state_dict()[k],v) for k,v in second.state_dict().items())
    edges=({k:v[:4] for k,v in a.items()},{k:v[4:8] for k,v in a.items()},np.ones((4,3),dtype='float32'))
    train(ordinary,Toy(),paid,device='cpu',steps=2)
    train(mutual,Toy(),paid,device='cpu',steps=2,mutual=True,edges=edges,weights=np.full((2,3),.5))
    changes=[]
    for first,second in zip(ordinary,mutual):
        delta=max(float((p-q).abs().max()) for p,q in zip(first.parameters(),second.parameters()))
        assert delta>0;changes.append(delta)
    identical={k:np.repeat(v[:1],2,axis=0) for k,v in a.items()}
    identical['memory'][1]+=1
    f=risk_floor(identical,np.array([0,0]),np.array([.1,.9]),False)[0]
    m=risk_floor(identical,np.array([0,0]),np.array([.1,.9]),True)[0]
    assert abs(f-math.log(2))<1e-12 and m<f
    # Full ordered anchor state is available to BOTH students.
    before={k:torch.as_tensor(v) for k,v in a.items()};after={k:v.clone() for k,v in before.items()}
    after['bags'][:,0,-6]+=1;after['bags'][:,1,-6]-=1
    state_effect=[]
    with torch.no_grad():
        for model in models('cpu'):
            effect=float((model(**before)-model(**after)).abs().max());assert effect>0;state_effect.append(effect)
    weights=np.full((2,3),.5);labels=[dict(factor_mode='WHOLE',yes_probability=p) for p in (.8,.2)]
    p=np.full((2,2,3),.5);p[0,:,0]=(.8,.2);p[1,:,0]=(.2,.8)
    updated,report=update_experts(weights,p,labels);assert updated[0,0]>.5 and np.array_equal(updated[:,1:],weights[:,1:])
    eps=.001;target=np.array([.8,.2]);mixture=(p[:,:,0]*weights[:,0,None]).sum(0)
    loss=lambda z:float((-target*np.log(z)-(1-target)*np.log1p(-z)).mean())
    assert loss(mixture)<=sum(weights[i,0]*loss(p[i,:,0]) for i in range(2))+1e-12
    real=[];use_verified_local_features()
    for case in verify_plan()['cases']:
        records=json.loads((PARENT/case['pair']/'paid_answers.json').read_text())['paid'];held=folds(records)
        assert len(set(sum(held,[])))==sum(map(len,held))
        future=set(sum(held,[]));training=[r for i,r in enumerate(records) if i not in future]
        assert {r['local_query'] for r in training if r['factor_mode']=='WHOLE'}=={r['local_query'] for r in records}
        withheld_groups={(records[i]['local_query'],records[i]['track_id'],records[i]['frame_id']) for i in future}
        assert not any((r['local_query'],r['track_id'],r['frame_id']) in withheld_groups for r in training)
        video=JointObservationVideo(case['pair'],case['video']);inputs=MemoryInputs(video)
        arrays=inputs.arrays(records[:3]);assert arrays['memory'].shape==(3,2202)
        assert all(np.isfinite(v).all() for v in arrays.values())
        left,right,weight=inputs.edge_data(np.array([0,len(inputs.left)*len(video.queries)-1]))
        assert ((weight>=0)&(weight<=1)).all() and np.all(weight[:,1]>=weight[:,2])
        real.append(dict(pair=case['pair'],heldout_rows=list(map(len,held)),initial_teacher_rows=len(training),
            physical_edges=len(inputs.left),memory_shape=list(inputs.memory.shape)))
    assert not guard
    receipt=dict(status='PASS_RECIPROCAL_GRADIENT_INPUT_AND_TEACHER_FOLD_CONTROLS',real_cases=real,
        both_students_differ_from_independent_teacher_training=changes,
        full_ordered_anchor_state_effect=state_effect,external_teacher_weights_update_correct_direction=True,
        current_and_future_probe_groups_not_trained=True,ordinary_architectures_initial_states_identical=True,
        exact_conflict_floor_changes_when_memory_makes_inputs_distinct=True,
        convex_teacher_logloss_mixture_check=True,no_HOTA_fits_evaluation_or_manual_GT=True,
        tiny_synthetic_fits_only=True,goal_status='ACTIVE')
    atomic_json(OUT/'contract.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
