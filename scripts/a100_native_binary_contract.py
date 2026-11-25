"""Real-input engineering and metadata-only budget audits, never task GT."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.native_binary_inputs import inventory,NativeVisual,BinaryBroker,prefix_arrays
from flex_uda.active.native_binary_models import make_model,log_probabilities
from flex_uda.active.native_binary_fit import coalesce,fit_prefix,risk,full_risk
from flex_uda.active.native_binary_curriculum import progress_reward,probe_measure,predict_pair
from flex_uda.active.rsi_discrimination_feedback import paired_probe_reservation
from scripts.a100_six_direction import OUT as SIX,PUBLIC


def budget():
    guard=generation_guard();rows=[]
    for pair,(_,target) in PAIRS.items():
        for video in sorted(metadata(target)['videos']):
            actions,quotas=inventory(pair,video);r=paired_probe_reservation(actions,quotas)
            rows.append(dict(pair=pair,video=video,actions=len(actions),budget=sum(quotas.values()),
                probe_rounds=r['fresh_probe_rounds'],paid_probe_requests=r['actual_probe_count']))
    assert len(rows)==148 and not guard
    report=dict(status='PASS_NATIVE_BINARY_ALL_SIX_METADATA_BUDGETS',settings_videos=len(rows),
        full_three_fresh_probe_rounds=sum(r['probe_rounds']==3 for r in rows),
        same_budget_per_query_and_all_queries_retained=True,
        teacher_answers_or_GPU_or_manual_GT_not_read=True,per_setting_video=rows)
    atomic_json(PUBLIC/'native_binary_budget_audit_20261124.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_setting_video'}),flush=True)


def engineer(device,fit):
    guard=generation_guard();torch.set_num_threads(2)
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    if device!='cpu':torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.15)
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    use_verified_local_features();video=sorted(metadata('Dance')['videos'])[0]
    v=JointObservationVideo('C1',video);actions,quota=inventory('C1',video);visual=NativeVisual(v,actions)
    reservation=paired_probe_reservation(actions,quota);broker=BinaryBroker('C1',video,actions,quota,reservation)
    counts={};uids=[]
    for uid,a in sorted(actions.items()):
        q,t=a['local_query'],a['local_track']
        if (q,t) in reservation['groups'] or counts.get(q,0)>=reservation['training_schedule'][0][q]:continue
        uids.append(uid);counts[q]=counts.get(q,0)+1
    paid=broker.request(uids,'train',0)
    if fit:
        for stage in range(3):
            if stage:
                stage_counts={};chosen=[]
                for uid,a in sorted(actions.items()):
                    q,t=a['local_query'],a['local_track']
                    if uid in broker.received or (q,t) in reservation['groups'] or stage_counts.get(q,0)>=reservation['training_schedule'][stage][q]:continue
                    chosen.append(uid);stage_counts[q]=stage_counts.get(q,0)+1
                paid.extend(broker.request(chosen,'train',stage))
            paid.extend(broker.request(reservation['probes'][stage],'probe',stage))
        broker.receipt(True)
    record=visual.record(paid)
    mean=visual.norm_sum/visual.norm_rows
    std=np.sqrt(np.maximum(visual.norm_square/visual.norm_rows-mean**2,0)).clip(min=.01)
    arrays=prefix_arrays([record],mean,std)
    recipes=[dict(family='structured',conditional=True,reference=True),dict(family='pair',risk='joint'),
        dict(family='pair',risk='marginal'),dict(family='point'),dict(family='coarse'),dict(family='point',risk='logit')]
    checks=[]
    for recipe in recipes:
        selected,projection=coalesce(arrays,recipe)
        data={k:torch.as_tensor(a,device=device) for k,a in selected.items()}
        torch.manual_seed(0);model=make_model(recipe,1101,64).to(device)
        p=log_probabilities(model,data,recipe)
        assert bool(torch.isfinite(p).all()) and float((p.exp().sum(-1)-1).abs().max())<1e-6
        # Actual objective and gradients agree under chunked accumulation.
        model.zero_grad();dense,_=risk(model,data,recipe);dense.backward()
        gradient=[None if x.grad is None else x.grad.detach().clone() for x in model.parameters()]
        model.zero_grad();streamed,_=full_risk(model,data,recipe,True,batch=7)
        gradient_error=max(float((a-x.grad).abs().max()) for a,x in zip(gradient,model.parameters()) if a is not None)
        assert abs(float(dense.detach()-streamed))<1e-6 and gradient_error<1e-5
        row=dict(recipe=recipe,shape=list(p.shape),projection=projection,
            streamed_gradient_max_error=gradient_error)
        if fit:
            _,_,_,receipt=fit_prefix(arrays,recipe,mean,std,device=device)
            row['fit']=receipt
            print(json.dumps(dict(recipe=recipe,excess_risk=receipt['excess_risk'],
                ordinary_gate=receipt['teacher_fit_gate_passed'])),flush=True)
        checks.append(row)
    synthetic={'best_query_view_constant_CE':.2,'conditional_teacher_information':.1}
    assert progress_reward(.4,.3,synthetic)['reward']==0
    assert progress_reward(.3,.3,synthetic)['reward']==0
    assert progress_reward(.19,.12,synthetic)['reward']>0
    if fit:
        lookup={r['uid']:r for r in paid};probe=[lookup[u] for u in reservation['probes'][0]]
    else:probe=broker.request(reservation['probes'][0],'probe',0)
    model=make_model(dict(family='structured',conditional=True,reference=True),1101,64).to(device)
    before,summary=probe_measure(model,probe,visual,mean,std,device)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.startswith(('count_encoder.','base.','interaction.')):p.add_(.01)
    after,again=probe_measure(model,probe,visual,mean,std,device)
    assert before==after and summary==again and progress_reward(before,after,summary)['reward']==0
    assert not guard
    report=dict(status='PASS_NATIVE_BINARY_REAL_INPUT_CONTRACT',pair='C1',video=video,
        metadata_actions=len(actions),paid_library_requests=len(paid),budget=sum(quota.values()),
        probe_rounds=reservation['fresh_probe_rounds'],record=record['report'],models=checks,
        new_teacher_GPU_calls=0,head_fits_executed=fit,device=device,GT_open_attempts=0,
        count_only_parameter_updates_have_exactly_zero_semantic_reward=True,
        execution=execution,
        no_RMOT_predictions_or_quality_claim=True)
    filename='all_paid_fit_contract' if fit else 'input_contract'
    atomic_json(SIX/'native_binary_engineering_20261124'/(filename+'.json'),report)
    atomic_json(PUBLIC/('native_binary_'+filename+'_20261124.json'),report)
    print(json.dumps(dict(status=report['status'],head_fits_executed=fit)),flush=True)


def decode(device):
    guard=generation_guard();torch.manual_seed(0);torch.set_num_threads(2)
    if device!='cpu':torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.15)
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from scripts.a100_native_binary_six_setting import RECIPES,METHODS,dense_scores
    use_verified_local_features();video=sorted(metadata('Dance')['videos'])[0]
    original=JointObservationVideo('C1',video);actions,_=inventory('C1',video);visual=NativeVisual(original,actions)
    frames=np.unique(original.obs[:,0]);rows=np.r_[np.flatnonzero(original.obs[:,0]==frames[0]),np.flatnonzero(original.obs[:,0]==frames[1])[:1]]
    class Slice:
        obs=original.obs[rows];columns=original.columns[rows];queries=original.queries[:1]
        raw_bags=original.raw_bags;present=original.present
        def dense_rows(self,q,i):return original.dense_rows(q,rows[i])
    sliced=Slice();mean=visual.norm_sum/visual.norm_rows
    std=np.sqrt(np.maximum(visual.norm_square/visual.norm_rows-mean**2,0)).clip(min=.01)
    models={}
    for name,recipe in RECIPES.items():
        model=make_model(recipe,1101,64).to(device).eval()
        with torch.no_grad():
            for p in model.parameters():p.add_(torch.randn_like(p)*.01)
        models[name]=(model,model)
    folder=SIX/'native_binary_engineering_20261124/decoded_slice';folder.mkdir(parents=True,exist_ok=True)
    scores=dense_scores(sliced,visual,models,mean,std,folder,device)
    assert set(scores)==set(METHODS) and all(np.isfinite(s).all() for s in scores.values())
    for name,recipe in RECIPES.items():
        if recipe['family']!='structured':
            assert np.max(np.abs(scores[name+'_FINAL']-scores[name+'_MEAN']))<1e-10
    assert not guard
    report=dict(status='PASS_NATIVE_BINARY_ALL_PATHS_REAL_DENSE_SLICE',raw_paths=len(scores),
        real_candidate_observations=len(rows),multiple_candidate_and_singleton_frames=True,
        final_equals_mean_for_identical_heads=True,all_scores_finite_float64=True,
        device=device,GT_open_attempts=0,no_RMOT_quality_claim=True)
    atomic_json(PUBLIC/'native_binary_decode_contract_20261124.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('budget','engineer','decode'),required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--fit',action='store_true');args=parser.parse_args()
    if args.mode=='budget':budget()
    elif args.mode=='engineer':engineer(args.device,args.fit)
    else:decode(args.device)
