"""Real-input S/R engineering closure before the complete primary study.

One metadata-first video in EACH of the six settings, two 200-step rounds.
This is deliberately not an RMOT quality screen: no masks/GT/HOTA/ablation.
All existing full-primary inputs and old sources are immutable. The goal
continues after this engineering prerequisite; it cannot establish three
contributions or superiority to sufficiently fitted teacher distillation.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

SUP = ROOT/'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
DEST = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/engineering'
PLAN = DEST/'registration.json'
SOURCES = ('scripts/a100_conditional_state_engineering.py',
           'flex_uda/active/conditional_state_learning.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    if PLAN.exists():raise FileExistsError(PLAN)
    original = json.loads((BASE/'registration.json').read_text())
    contract = SUP/'conditional_state_contract_20261127/result.json'
    assert json.loads(contract.read_text())['status'].startswith('PASS_')
    cases, seen, paths = [], set(), {BASE/'registration.json', BASE/'all4588_sealed_before_GT.json', contract}
    for case in original['cases']:
        if case['pair'] in seen:continue
        seen.add(case['pair']);case = dict(case)
        parent = BASE/'units'/case['pair']/case['evaluated_video']
        case['parent'] = str(parent)
        case['ordinary_model'] = str(parent/'ordinary/UNIFORM_SOFT_KD288/UNIFORM_SOFT_KD.pt')
        case['original_scores'] = str(parent/'predictions/UNIFORM_SOFT_KD288/scores.npy')
        case['uniform_answers'] = str(parent/'uniform_answers.jsonl')
        case['uniform_draws'] = str(parent/'uniform_draws.json')
        cases.append(case)
        paths.update(Path(case[k]) for k in ('ordinary_model','original_scores','uniform_answers','uniform_draws',
            'observations','temporal_queries','auxiliary_path'))
        paths.add(parent/'unlabeled_normalizer.npz');paths.add(parent/'generation_complete.json')
    assert len(cases) == 6 and seen == {'C1','C2','C3','C4','C5','C6'}
    for name, digest in original['source_sha256'].items():
        assert file_sha(ROOT/name) == digest, name
    plan = dict(status='REGISTERED_BEFORE_REAL_INPUT_NEURAL_ENGINEERING', cases=cases,
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},
        original42_source_sha256=original['source_sha256'],
        input_sha256={str(p):file_sha(p) for p in sorted(paths)},
        GPUs=[0,1,2,3], seed=0, memory_fraction=.20, protect_existing_GPU2_service=True,
        teacher='Original frozen Qwen; replay only the already-paid object-balanced uniform event set. No new Qwen forward or weight update.',
        solver='Full original9352D PlainHead288, initialized from that unit\'s adequate uniform soft KD. Engineering steps are not sufficient primary training.',
        observer='Eight observable change covariates ->32hidden ->copy/reset gate plus three oriented sensitivity/specificity pairs.',
        objective='Negative query-and-per-query-budget weighted structured teacher-decision ELBO + .25 mean AUX soft BCE + .001 mean Bernoulli KL(clamp(base_keep,.001,.999) || keep) + .01 mean Bernoulli KL(.8 || channel).',
        latent_model='Copy previous referring state or redraw from current neural solver. CHECKLIST video-existence mixture; PRESUPPOSED fixes video existence, never forces a candidate/frame match. Same protocol and memory processing must be given to ordinary controls in the actual quality study.',
        observation_scope='CLIP has a uniform latent time over DISTINCT actual provided input observations. Duplicate input crops retained in provenance, not treated as independent votes. This is an unverified observation model, not a claim that Qwen computes any/last/majority truth.',
        protocol='Existing v8P query-set heuristic using task text metadata only; explicitly DEV, not an externally supplied contract or a GT-derived domain prior.',
        rounds=2, solver_steps_per_round=200, observer_steps_per_round=200, learning_rate=.0001,
        unlabeled_MC_nodes_per_round=8192, include_each_observed_anchor=True,
        MC='Balance-heuristic estimator: uniform sample multiplicity plus each paid anchor once, weighted by inverse expected multiplicity Nsample/Nnodes + I_anchor. Normalize using FULL frozen-E-step mass, not sampled target counts.',
        fixed_backtracking_fractions=[1.,.5,.25,.125,.0625,.03125,0.],
        acceptance='Recompute full real-input objective; only finite nonincrease accepted, otherwise restore parameters. This is an optimization check, not semantic or HOTA selection.',
        reliability_orientation='Every sensitivity/specificity in(.5,.99) is a modeling assumption; one noisy teacher does not identify semantic truth. Repeated-time errors may remain correlated.',
        not_a_primary_quality_screen=True, no_target_GT=True, no_predictions_masks_or_HOTA=True,
        no_new_seeds_or_module_ablation=True, old_numeric_sources_unchanged=True,
        future_primary_required='All148 units, adequate soft/hard/same-architecture/fused KD and track means, matched protocol/fast readouts, all outputs sealed before isolated GT evaluation, whole-setting raw and dual CountMatched, complete cost accounting.',
        goal_status='ACTIVE')
    assert not guard;atomic_json(PLAN,plan)
    print(json.dumps(dict(status=plan['status'],cases=len(cases),inputs=len(paths))),flush=True)


def read_plan():
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','original42_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    for name,sha in plan['input_sha256'].items():assert file_sha(name)==sha,name
    return plan


def source_events(context, case):
    import numpy as np
    from scripts.active_run import video_spec
    from flex_uda.active.conditional_state_learning import ScopeEvent,VIEWS
    video=context.video;delta=video_spec(case['target'],case['evaluated_video'],video.bank)['delta']
    replies=[json.loads(line) for line in Path(case['uniform_answers']).read_text().splitlines()]
    draws=json.loads(Path(case['uniform_draws']).read_text())['draws']
    by_uid={row['uid']:row for row in replies};assert len(by_uid)==len(replies)==len(draws)
    tracks={int(t):np.flatnonzero(video.obs[:,1]==t) for t in np.unique(video.obs[:,1])}
    events=[];provenance=[]
    for draw in draws:
        uid=draw['uid'];row=by_uid[uid];q,node=divmod(int(uid),context.n)
        assert float(row['probability'])==float(draw['probability'])
        assert row['frozen_weight_sha256']=='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
        frame,track=map(int,video.obs[node]);view=row['effective_view'];nodes=[node]
        input_nodes=[node]
        if view=='CLIP':
            candidates=tracks[track];times=video.obs[candidates,0];input_nodes=[]
            for offset in (3*delta,2*delta,delta,0):
                index=int(np.searchsorted(times,frame-offset,side='right'))-1
                if index>=0:input_nodes.append(int(candidates[index]))
            assert len(input_nodes)>=2 and input_nodes[-1]==node
            nodes=sorted(set(input_nodes),key=lambda i:video.obs[i,0])
        event=ScopeEvent(q,tuple(nodes),int(float(row['probability'])>=.5),VIEWS.index(view),uid)
        events.append(event);provenance.append(dict(uid=uid,requested_view=row['requested_view'],
            effective_view=view,actual_input_observation_nodes=input_nodes,distinct_scope_nodes=nodes,
            original_soft_probability=float(row['probability']),decision=event.answer))
    return events,provenance


def unit(case,gpu,plan):
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
    folder=DEST/case['pair'];folder.mkdir(exist_ok=True,parents=True)
    if (folder/'completion.json').exists():return json.loads((folder/'completion.json').read_text())
    started=time.perf_counter();shutil.copyfile(Path(case['parent'])/'unlabeled_normalizer.npz',folder/'unlabeled_normalizer.npz')
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
    for stage in range(plan['rounds']):
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
        for step in range(plan['solver_steps_per_round']):
            at=(cursor+step*512)%len(x);optimizer.zero_grad(set_to_none=True);z=solver(x[at])
            loss=((nn.functional.softplus(z)-y[at]*z)*w[at]*len(x)).mean()
            loss=loss+.25/full_mass*nn.functional.binary_cross_entropy_with_logits(solver(aux_x),aux_y)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(solver.parameters(),10);optimizer.step()
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
        rounds.append(row);atomic_json(folder/f'round{stage}.json',row)
        torch.save(dict(solver=state(solver),observer=state(observer),rounds=rounds),folder/'latest.pt')
        print(json.dumps(dict(status='REAL_CONDITIONAL_STATE_ENGINEERING_ROUND',pair=case['pair'],
            GPU=gpu,stage=stage,objective_before=before,objective_after=after,
            accepted_fraction=fraction,parameter_L2_change=deltas)),flush=True)
    receipt=dict(status='COMPLETE_REAL_INPUT_CONDITIONAL_STATE_ENGINEERING',pair=case['pair'],GPU=gpu,
        physical_observations=context.n,queries=context.nq,events=len(events),protocol=protocol,
        initial_KD_reconstruction_max_error=reconstruction_error,rounds=rounds,
        wall_seconds=time.perf_counter()-started,GT_opens=0,new_teacher_calls=0,new_masks=0,
        HOTA_evaluations=0,not_sufficient_primary_training=True,not_RSI_causal_quality_evidence=True,
        teacher_inputs_and_features_prepaid_not_fresh_cost=True)
    atomic_json(folder/'completion.json',receipt);return receipt


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    plan=read_plan();result=[]
    for i,case in enumerate(plan['cases']):
        if i%4==gpu:result.append(unit(case,gpu,plan));torch.cuda.empty_cache()
    assert not guard
    atomic_json(DEST/f'GPU{gpu}_completion.json',dict(status='COMPLETE_ENGINEERING_WORKER',
        GPU=gpu,execution=execution,units=[r['pair'] for r in result],GT_open_attempts=guard))


def main():
    plan=read_plan();processes=[]
    for gpu in (0,1,2,3):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        log=(DEST/f'GPU{gpu}.log').open('a',buffering=1)
        proc=subprocess.Popen([sys.executable,__file__,'--worker',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((gpu,proc,log));print(json.dumps(dict(status='REAL_ENGINEERING_WORKER_STARTED',GPU=gpu,pid=proc.pid)),flush=True)
    failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(proc.wait):(gpu,log) for gpu,proc,log in processes}
        for future in concurrent.futures.as_completed(futures):
            gpu,log=futures[future];code=future.result();log.close()
            print(json.dumps(dict(status='REAL_ENGINEERING_WORKER_EXIT',GPU=gpu,exit_code=code)),flush=True)
            if code:failed.append(dict(GPU=gpu,exit_code=code))
    for group in ('source_sha256','original42_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    if failed:
        atomic_json(DEST/'failure.json',dict(status='ENGINEERING_FAILURE_PRESERVED',failed=failed));raise RuntimeError(failed)
    receipts=[json.loads((DEST/c['pair']/'completion.json').read_text()) for c in plan['cases']]
    atomic_json(DEST/'completion.json',dict(status='COMPLETE_ALL6_SETTING_REAL_INPUT_ENGINEERING',cases=receipts,
        source_sha256=plan['source_sha256'],no_GT_teacher_or_HOTA=True,goal_complete=False,
        next='Implement/register the complete adequately trained six-setting primary and matched controls; no module ablation matrix.'))
    print(json.dumps(dict(status='COMPLETE_ALL6_SETTING_REAL_INPUT_ENGINEERING',units=6,goal_complete=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker is not None:worker(args.worker)
    else:main()
