"""Full-scope conditional state training with atomic round recovery.

The numerical update is copied from the successful registered engineering
kernel. New behavior is explicit folders, fixed longer training, phase costs
and atomic accepted-state/RNG recovery. Old pinned sources are unchanged.
An unfinished round restarts from its last COMPLETE round, not partial weights;
all attempted work and failed worker receipts are kept separately.
"""
import json
import os
from pathlib import Path
import shutil
import time
ROOT=Path(__file__).resolve().parents[2]
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_conditional_state_engineering import source_events


def save_round(path,value):
    import torch
    path=Path(path)
    temporary=path.with_name(path.name+'.atomic.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def load_round(path,registration_sha256):
    import numpy as np
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['registration_sha256']==registration_sha256,'Checkpoint belongs to a different registration'
    assert [r['stage'] for r in saved['rounds']]==list(range(len(saved['rounds'])))
    assert np.isfinite(saved['accepted_scores']).all()
    for row in saved['rounds']:
        assert row['objective_after']<=row['objective_before']+1e-9
        assert row['solver_attempted_steps']>0 and row['observer_attempted_steps']>0
    return saved


def learn(case,gpu,plan,folder):
    import numpy as np
    import torch
    from scipy.special import expit
    from torch import nn
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active import evidence_visual_learning as core
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.conditional_state_learning import (
        StateObserver,scoped_bound,observable_changes,accepted_fraction)
    from flex_uda.active.acquisition import expression_types
    from flex_uda.active.v8p import classify_protocol
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    if (folder/'completion.json').exists():
        done=json.loads((folder/'completion.json').read_text())
        assert done['registration_sha256']==plan['registration_sha256']
        assert done['checkpoint_sha256']==file_sha(folder/'latest.pt')
        return done
    started=time.perf_counter()
    atomic_json(folder/f'attempt_{os.getpid()}.json',dict(status='STARTED',time_unix=time.time(),registration_sha256=plan['registration_sha256']))
    shutil.copyfile(Path(case['parent'])/'unlabeled_normalizer.npz',folder/'unlabeled_normalizer.npz')
    context=Context(case,folder,None,gpu);video=context.video;N=context.n*context.nq
    old=torch.load(case['ordinary_model'],map_location='cpu',weights_only=False)
    solver=core.make(context.mean,context.std,'cuda:0',kind='plain',width=288)
    solver.load_state_dict(old['model']);solver.eval()
    initial=np.load(case['original_scores']).astype(float);assert initial.shape==(context.nq,context.n)
    check_rows=np.unique(np.linspace(0,N-1,min(512,N)).astype(int))
    actual=core.logits(solver,context.features[check_rows],'cuda:0')
    reconstruction_error=float(np.max(np.abs(actual-initial.reshape(-1)[check_rows])))
    assert reconstruction_error<2e-5,reconstruction_error
    events,provenance=source_events(context,case)
    atomic_json(folder/'teacher_event_provenance.json',dict(events=provenance))
    md=metadata(case['target']);sets={v:[q['query_text'] for q in md['episodes'] if q['video_id']==v] for v in md['videos']}
    protocol=classify_protocol(sets)['protocol']
    dynamic=np.array([bool(set(expression_types(q['query_text']))&{'MOTION','SPATIAL','ROLE'}) for q in video.queries])
    change=np.empty((context.nq,context.n,8),dtype='float32');base=np.empty((context.nq,context.n),dtype='float32')
    for q in range(context.nq):
        pcd=video._pcd_rows(np.full(context.n,q),np.arange(context.n))
        c,b,_=observable_changes(video.obs,context.images,pcd[None],video.geometry[:,:4],dynamic[q:q+1])
        change[q],base[q]=c[0],b[0]
    np.savez(folder/'memory_inputs.npz',change=change,base=base)
    observer=StateObserver().to('cuda:0')
    flat_change=torch.as_tensor(change.reshape(-1,8),device='cuda:0')
    flat_base=torch.as_tensor(base.reshape(-1),device='cuda:0')
    aux_x=torch.as_tensor(context.auxiliary,device='cuda:0');aux_y=torch.as_tensor(context.ay,device='cuda:0')
    budget=np.bincount([e.query for e in events],minlength=context.nq);assert (budget>0).all()
    qw=1/(context.nq*budget)
    anchor=np.array([int(e.uid) for e in events]);assert len(set(anchor))==len(anchor)
    is_anchor=np.zeros(N,dtype=bool);is_anchor[anchor]=True
    generator=np.random.default_rng(0)

    def decode(scores):
        with torch.no_grad():
            keep=observer(flat_change,flat_base).cpu().numpy().reshape(base.shape)
            rel=observer.reliability().cpu().numpy()
            aux=float(nn.functional.binary_cross_entropy_with_logits(solver(aux_x),aux_y))
        expected=scoped_bound(expit(scores),keep,video.obs,events,rel,protocol=protocol,existence_prior=.5,iterations=8)
        keep_clip=np.clip(keep,1e-9,1-1e-9);b=np.clip(base,.001,.999)
        regular=float(np.mean(b*np.log(b/keep_clip)+(1-b)*np.log((1-b)/(1-keep_clip))))
        channel=float(np.mean(.8*np.log(.8/rel)+.2*np.log(.2/(1-rel))))
        objective=-float(np.sum(qw*expected['bound']))+.25*aux+.001*regular+.01*channel
        return expected,objective,keep,rel

    def full_scores():
        output=np.empty(N,dtype='float32')
        for begin in range(0,N,512):
            rows=np.arange(begin,min(begin+512,N));output[rows]=core.logits(solver,context.features[rows],'cuda:0')
        return output.reshape(context.nq,context.n)

    def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    scores=initial;rounds=[]
    if (folder/'latest.pt').exists():
        saved=load_round(folder/'latest.pt',plan['registration_sha256'])
        solver.load_state_dict(saved['solver']);observer.load_state_dict(saved['observer'])
        scores=saved['accepted_scores'];rounds=saved['rounds']
        assert scores.shape==initial.shape and len(rounds)<=plan['rounds']
        generator.bit_generator.state=saved['generator_state']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        print(json.dumps(dict(status='RESTORED_CONDITIONAL_STATE_ROUND',pair=case['pair'],video=case['evaluated_video'],completed=len(rounds))),flush=True)
    for stage in range(len(rounds),plan['rounds']):
        round_started=time.perf_counter()
        expected,before,keep,rel=decode(scores)
        old_s,old_r=state(solver),state(observer)
        sample=generator.integers(N,size=plan['unlabeled_MC_nodes_per_round'])
        ids=np.r_[sample,anchor]
        multiplicity=len(sample)/N+is_anchor[ids]
        mass=expected['reset_positive']+expected['reset_negative']
        target=np.divide(expected['reset_positive'].reshape(-1)[ids],mass.reshape(-1)[ids],
                         out=np.zeros(len(ids)),where=mass.reshape(-1)[ids]>1e-20)
        weights=(mass*qw[:,None]).reshape(-1)[ids]/multiplicity
        full_mass=float(np.sum(mass*qw[:,None]))+.25
        x=torch.as_tensor(context.features[ids],device='cuda:0');y=torch.as_tensor(target,dtype=torch.float32,device='cuda:0')
        w=torch.as_tensor(weights/full_mass,dtype=torch.float32,device='cuda:0')
        optimizer=torch.optim.Adam(solver.parameters(),lr=plan['learning_rate'])
        cursor=torch.arange(512,device='cuda:0');solver.train()
        solver_started=time.perf_counter()
        for step in range(plan['solver_steps_per_round']):
            at=(cursor+step*512)%len(x);optimizer.zero_grad(set_to_none=True);z=solver(x[at])
            loss=((nn.functional.softplus(z)-y[at]*z)*w[at]*len(x)).mean()
            loss=loss+.25/full_mass*nn.functional.binary_cross_entropy_with_logits(solver(aux_x),aux_y)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(solver.parameters(),10);optimizer.step()
        torch.cuda.synchronize();solver_seconds=time.perf_counter()-solver_started
        del x,y,w;solver.eval()
        # Same frozen E-step supplies both neural M steps, not peer agreement
        # treated as external truth. Each teacher event enters the channel once.
        copy=torch.as_tensor(expected['copy'].reshape(-1)[ids],dtype=torch.float32,device='cuda:0')
        survival=torch.as_tensor((expected['copy']+mass).reshape(-1)[ids],dtype=torch.float32,device='cuda:0')
        qw_sample=np.repeat(qw,context.n)[ids]/multiplicity
        rw=torch.as_tensor(qw_sample,dtype=torch.float32,device='cuda:0')
        prior_weight=torch.as_tensor(1/(N*multiplicity),dtype=torch.float32,device='cuda:0')
        rx,rb=flat_change[ids],flat_base[ids]
        valid=torch.as_tensor(np.tile(expected['layout']['previous']>=0,context.nq)[ids],device='cuda:0')
        truth=np.array([expected['existence'][e.query]*np.dot(resp,expected['conditional_posterior'][e.query,list(e.nodes)])
                        for e,resp in zip(events,expected['responsibilities'])])
        truth=torch.tensor(truth,dtype=torch.float32,device='cuda:0')
        answer=torch.tensor([e.answer for e in events],dtype=torch.float32,device='cuda:0')
        views=torch.tensor([e.view for e in events],device='cuda:0')
        ew=torch.tensor([qw[e.query] for e in events],dtype=torch.float32,device='cuda:0')
        optimizer_r=torch.optim.Adam(observer.parameters(),lr=plan['learning_rate'])
        r_mass=max(float(np.sum((expected['copy']+mass)[:,expected['layout']['previous']>=0]*qw[:,None])),1e-12)
        observer_started=time.perf_counter()
        for step in range(plan['observer_steps_per_round']):
            at=(cursor+step*512)%len(ids);optimizer_r.zero_grad(set_to_none=True)
            rr=observer(rx[at],rb[at]).clamp(1e-6,1-1e-6)
            cost=-(copy[at]*rr.log()+(survival[at]-copy[at])*torch.log1p(-rr))
            cost=(cost*rw[at]*valid[at]*len(ids)).mean()/r_mass
            ch=observer.reliability();s,c=ch[views,0],ch[views,1]
            l1=answer*s.log()+(1-answer)*torch.log1p(-s)
            l0=answer*torch.log1p(-c)+(1-answer)*c.log()
            cost=cost-((truth*l1+(1-truth)*l0)*ew).sum()/r_mass
            cost=cost+.01/r_mass*(.8*torch.log(.8/ch)+.2*torch.log(.2/(1-ch))).mean()
            # A shared prior resists ungrounded changes in the writer. This is
            # not a claim that initial hazards are semantically correct.
            bv=rb[at].clamp(.001,.999)
            prior_cost=bv*torch.log(bv/rr)+(1-bv)*torch.log((1-bv)/(1-rr))
            cost=cost+.001/r_mass*(prior_cost*prior_weight[at]*len(ids)).mean()
            assert torch.isfinite(cost);cost.backward();nn.utils.clip_grad_norm_(observer.parameters(),10);optimizer_r.step()
        torch.cuda.synchronize();observer_seconds=time.perf_counter()-observer_started
        proposed_s,proposed_r=state(solver),state(observer)
        candidate_records=[]

        def evaluate(fraction):
            for model,old,new in ((solver,old_s,proposed_s),(observer,old_r,proposed_r)):
                model.load_state_dict({key:old[key]+fraction*(new[key]-old[key]) if old[key].is_floating_point()
                                       else old[key] for key in old})
            candidate=full_scores();_,objective,_,_=decode(candidate)
            candidate_records.append(dict(fraction=fraction,objective=objective))
            np.save(folder/'latest_candidate_scores.npy',candidate)
            return objective

        fraction,trials=accepted_fraction(before,evaluate)
        if fraction:
            scores=np.load(folder/'latest_candidate_scores.npy')
        else:
            solver.load_state_dict(old_s);observer.load_state_dict(old_r)
        _,after,_,newrel=decode(scores)
        assert after<=before+1e-9,(before,after)
        deltas={}
        for name,model,old in [('solver',solver,old_s),('observer',observer,old_r)]:
            deltas[name]=float(sum(((value.detach().cpu()-old[key]).double().square().sum())
                                  for key,value in model.state_dict().items() if value.is_floating_point()).sqrt())
        row=dict(stage=stage,objective_before=before,objective_after=after,accepted_fraction=fraction,
                 line_search=trials,parameter_L2_change=deltas,channel=newrel.tolist(),
                 protocol=protocol,mc_rows=len(sample),paid_anchor_rows=len(anchor),
                 all_provided_observations=context.n,all_queries=context.nq)
        row.update(solver_attempted_steps=plan['solver_steps_per_round'],observer_attempted_steps=plan['observer_steps_per_round'],solver_seconds=solver_seconds,observer_seconds=observer_seconds,round_wall_seconds=time.perf_counter()-round_started)
        rounds.append(row);atomic_json(folder/f'round{stage}.json',row)
        save_round(folder/'latest.pt',dict(solver=state(solver),observer=state(observer),rounds=rounds,accepted_scores=np.asarray(scores).copy(),generator_state=generator.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),registration_sha256=plan['registration_sha256']))
        print(json.dumps(dict(status='FULL_CONDITIONAL_STATE_LEARNING_ROUND',pair=case['pair'],
            GPU=gpu,stage=stage,objective_before=before,objective_after=after,
            accepted_fraction=fraction,parameter_L2_change=deltas)),flush=True)
    receipt=dict(status='COMPLETE_REGISTERED_CONDITIONAL_STATE_LEARNING',pair=case['pair'],GPU=gpu,
        physical_observations=context.n,queries=context.nq,events=len(events),protocol=protocol,
        initial_KD_reconstruction_max_error=reconstruction_error,rounds=rounds,
        wall_seconds=time.perf_counter()-started,GT_opens=0,new_teacher_calls=0,new_masks=0,
        HOTA_evaluations=0,registered_rounds_complete=True,not_RSI_causal_quality_evidence=True,
        registration_sha256=plan['registration_sha256'],checkpoint_sha256=file_sha(folder/'latest.pt'),memory_inputs_sha256=file_sha(folder/'memory_inputs.npz'),
        solver_attempted_additional_steps=sum(x['solver_attempted_steps'] for x in rounds),observer_attempted_steps=sum(x['observer_attempted_steps'] for x in rounds),
        teacher_inputs_and_features_prepaid_not_fresh_cost=True)
    atomic_json(folder/'completion.json',receipt)
    atomic_json(folder/f'attempt_{os.getpid()}.json',dict(status='COMPLETE',wall_seconds=time.perf_counter()-started,registration_sha256=plan['registration_sha256']))
    return receipt
