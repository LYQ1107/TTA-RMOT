"""A learned observer and object/state solver with paid, fresh binary probes."""
from collections import defaultdict
import copy
import math
import numpy as np
import torch
from flex_uda.active.native_binary_inputs import BinaryBroker,prefix_arrays
from flex_uda.active.native_binary_models import make_model,renormalize,log_probabilities,pair_marginal_logs
from flex_uda.active.native_binary_fit import fit_prefix
from flex_uda.active.rsi_curriculum import ProgressChallenger,challenger_features,select_round,update_challenger
from flex_uda.active.rsi_discrimination_feedback import paired_probe_reservation


def normalize_pair(raw,mean,std):
    result={k:(a.copy() if a.dtype==bool else ((a-mean)/std).astype('float32')) for k,a in raw.items()}
    result['anchors_A'][~result['present_A']]=0;result['anchors_B'][~result['present_B']]=0
    return result


def predict_pair(model,raw,mean,std,device,batch=1024):
    result=[]
    with torch.no_grad():
        for first in range(0,len(raw['first']),batch):
            normalized=normalize_pair({k:a[first:first+batch] for k,a in raw.items()},mean,std)
            data={k:torch.as_tensor(a,device=device) for k,a in normalized.items()}
            result.append(log_probabilities(model,data,dict(family='structured')).cpu().numpy())
    return np.concatenate(result)


def probe_measure(model,probe,visual,mean,std,device):
    if not probe:return None,None
    assert len(probe)%2==0
    points=visual.point_arrays([r['uid'] for r in probe])
    anchors=((points['point_anchors']-mean)/std).astype('float32')
    anchors[~points['point_present']]=0
    current=((points['point_current']-mean)/std).astype('float32')
    with torch.no_grad():
        utility=model.utility(torch.as_tensor(anchors,device=device),
            torch.as_tensor(points['point_present'],device=device),torch.as_tensor(current,device=device)).double()
        log_relative=torch.log_softmax(torch.stack((utility[::2],utility[1::2]),-1),-1).cpu().numpy()
    groups=defaultdict(list)
    for i,(a,b) in enumerate(zip(probe[::2],probe[1::2])):
        assert a['local_query']==b['local_query'] and a['case']['view']==b['case']['view']
        assert a['local_track']!=b['local_track']
        p,q=a['probability'],b['probability']
        mass=p*(1-q)+q*(1-p)
        relative=p*(1-q)/mass if mass else .5
        target=np.asarray([relative,1-relative])
        ce=-float((target*log_relative[i]).sum())*mass
        entropy=-float((target*np.log(target.clip(min=1e-300))).sum())*mass
        # AB and BA are both scored analytically from the SAME two paid
        # binary replies. A position/query/view-only predictor has optimum
        # relative probability .5, regardless of the absolute No rate.
        groups[a['local_query']].append((ce,mass*np.log(2.),entropy))
    values=np.asarray([np.mean(rows,axis=0) for rows in groups.values()])
    ce,null,entropy=values.mean(0)
    return float(ce),dict(best_query_view_constant_CE=float(null),
        teacher_entropy_floor=float(entropy),conditional_teacher_information=max(float(null-entropy),0.),
        probe_queries=len(groups),paid_probe_nodes=len(probe),manual_GT_used=False,
        semantic_utility_only_count_parameters_not_scored=True,
        both_pair_orientations_scored_without_additional_teacher_calls=True,
        risk_units='exactly-one-mass-weighted conditional pair CE, macro query')


def progress_reward(before,after,summary):
    if summary is None:return dict(reward=0.,skill_before=None,skill_after=None,information=None)
    null=summary['best_query_view_constant_CE'];information=summary['conditional_teacher_information']
    a,b=max(null-before,0.),max(null-after,0.)
    assert a<=information+1e-8 and b<=information+1e-8
    # Normalize by available teacher information only above the common .005
    # risk-resolution threshold. Tiny No-rate differences earn no credit.
    reward=math.tanh((b-a)/information) if information>=.005 else 0.
    return dict(reward=reward,skill_before=a,skill_after=b,information=information,
        information_resolution=.005,fresh_paid_teacher_risk_not_RMOT_truth=True)


def initial_model(recipe,dimension,mean,std,previous,device):
    torch.manual_seed(0);width=previous['width'] if previous else 64
    model=make_model(recipe,dimension,width).to(device).eval()
    if previous:
        assert previous['recipe']==recipe;model.load_state_dict(previous['model'])
        renormalize(model,recipe,previous['mean'],previous['std'],mean,std)
    return model


def observe_and_solve(visual,actions,quotas,mean,std,*,past=(),previous=None,actor_state=None,
                      learned=True,device='cuda:0',callback=None,resume=None):
    reservation=paired_probe_reservation(actions,quotas)
    broker=BinaryBroker(visual.video.pair,visual.video.video,actions,quotas,reservation)
    recipe=dict(family='structured',conditional=True,reference=True,identity=True,policy='LEARNED' if learned else 'FROZEN')
    if resume:previous,actor_state=resume['solver'],resume['actor']
    model=initial_model(recipe,visual.video.dimension,mean,std,previous,device)
    order=sorted(actions);raw=visual.arrays(order)
    feature=challenger_features(predict_pair(model,raw,mean,std,device),raw,actions,[])
    torch.manual_seed(0);actor=ProgressChallenger(feature.shape[-1]).cpu()
    generator=torch.Generator().manual_seed(0);baseline=0.
    optimizer=torch.optim.SGD(actor.parameters(),lr=.1,momentum=0.)
    if actor_state:
        actor.load_state_dict(actor_state['model']);generator.set_state(actor_state['sampling_generator'])
        optimizer.load_state_dict(actor_state['optimizer']);baseline=actor_state['baseline']
    train=[];paid=[];rounds=[];first_stage=0
    if resume:
        restored={}
        for event in resume['sequence']:
            rows=broker.request([event['uid']],event['role'],event['stage']);restored[event['uid']]=rows[0]
        train=[restored[u] for u in resume['training_UIDs']];paid=[restored[u] for u in resume['paid_UIDs']]
        rounds=resume['rounds'];first_stage=resume['stage']+1
    for stage in range(first_stage,3):
        logq=predict_pair(model,raw,mean,std,device)
        feature=challenger_features(logq,raw,actions,[r['uid'] for r in paid])
        chosen,log_policy=select_round(actor,feature,actions,quotas,reservation,[r['uid'] for r in paid],stage,generator)
        rows=broker.request(chosen,'train',stage);train.extend(rows);paid.extend(rows)
        probe=broker.request(reservation['probes'][stage],'probe',stage);paid.extend(probe)
        before,summary=probe_measure(model,probe,visual,mean,std,device)
        assert not({(r['local_query'],r['local_track']) for r in train}&reservation['groups'])
        record=visual.record(train)
        model,_,checkpoint,fit=fit_prefix(prefix_arrays([*past,record],mean,std),recipe,mean,std,previous,device)
        previous=checkpoint
        after,summary_after=probe_measure(model,probe,visual,mean,std,device)
        assert summary==summary_after
        credit=progress_reward(before,after,summary)
        update=update_challenger(actor,optimizer,log_policy,before,after,baseline,
            frozen=not learned,reward_override=credit['reward'])
        baseline=update['baseline_after']
        rounds.append(dict(stage=stage,fit=fit,actor_update=update,credit=credit,teacher_probe=summary,
            budget=broker.receipt(),training_nodes=len(train)))
        state=dict(model={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
            sampling_generator=generator.get_state(),optimizer=optimizer.state_dict(),baseline=baseline)
        if callback:callback(stage,dict(solver=checkpoint,actor=state,sequence=list(broker.sequence),
            training_UIDs=[r['uid'] for r in train],paid_UIDs=[r['uid'] for r in paid],rounds=rounds,stage=stage))
    record=visual.record(paid)
    model,early,checkpoint,fit=fit_prefix(prefix_arrays([*past,record],mean,std),recipe,mean,std,previous,device)
    state=dict(model={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
        sampling_generator=generator.get_state(),optimizer=optimizer.state_dict(),baseline=baseline)
    report=dict(rounds=rounds,final_all_paid_fit=fit,teacher_budget=broker.receipt(True),
        fresh_probe_rounds=reservation['fresh_probe_rounds'],probe_groups=len(reservation['groups']),
        actor_actual_updates=sum(r['actor_update']['actor_parameter_L2_change']>0 for r in rounds),
        entire_budget_includes_probes=True,current_video_transductive=True,GT_or_RMOT_metric_used=False,
        observer_and_solver_both_learned=learned,teacher_itself_frozen=True)
    return dict(model=model,early=early,checkpoint=checkpoint,actor=state,record=record,
        paid=paid,sequence=broker.sequence,report=report)


def replay_solver(visual,paid,sequence,recipe,mean,std,*,past=(),previous=None,device='cuda:0',scheduled=True):
    lookup={r['uid']:r for r in paid};train=[];receipts=[]
    if scheduled:
        for stage in range(3):
            train.extend(lookup[e['uid']] for e in sequence if e['stage']==stage and e['role']=='train')
            record=visual.record(train)
            _,_,previous,fit=fit_prefix(prefix_arrays([*past,record],mean,std),recipe,mean,std,previous,device)
            receipts.append(fit)
    record=visual.record(paid)
    model,early,checkpoint,fit=fit_prefix(prefix_arrays([*past,record],mean,std),recipe,mean,std,previous,device)
    receipts.append(fit)
    return model,early,checkpoint,dict(fits=receipts,same_all_paid_information=True,
        same_three_round_training_prefixes=scheduled,GT_or_task_metric_selection=False)
