"""Group-disjoint visual feedback for a correlated teacher-opinion model.

The split prevents direct teacher-example fitting by the predictor that supplies
that example's image prior. Global observer updates still couple the folds;
this is not a claim of statistical independence or semantic correctness.
"""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.joint_teacher_noise import make_solver,predict,fit_solver
from flex_uda.active.semantic_teacher_patterns import SemanticPatterns


def group_folds(keys):
    groups=[(str(row[0]),int(row[2])) for row in keys]
    counts=Counter(groups)
    assert len(counts)>=2
    order=sorted(counts,key=lambda g:(-counts[g],hashlib.sha256(json.dumps(g).encode()).hexdigest()))
    loads=[0,0];assigned={}
    for group in order:
        fold=int(loads[1]<loads[0])
        assigned[group]=fold;loads[fold]+=counts[group]
    result=np.asarray([assigned[g] for g in groups],dtype='int8')
    assert set(result)=={0,1}
    return result


def outside_predictions(models,features,folds,device):
    result=np.empty(len(features),dtype='float64')
    for trained_fold,model in enumerate(models):
        outside=folds!=trained_fold
        result[outside]=predict(model,features[outside],device)
    return result


def accept_pair(models,previous,observer,probability,features,folds,device):
    proposed=[copy.deepcopy(model.state_dict()) for model in models]
    for model,state in zip(models,previous): model.load_state_dict(state)
    before=observer.objective(probability,outside_predictions(models,features,folds,device))
    accepted,after=0.,before
    for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625):
        for model,old,new in zip(models,previous,proposed):
            model.load_state_dict({k:old[k]+fraction*(new[k]-old[k]) for k in old})
        value=observer.objective(probability,outside_predictions(models,features,folds,device))
        if np.isfinite(value) and value>=before-1e-7:
            accepted,after=fraction,value
            break
    if not accepted:
        for model,state in zip(models,previous): model.load_state_dict(state)
    changes=[max(float((v-old[k]).abs().max()) for k,v in model.state_dict().items())
             for model,old in zip(models,previous)]
    return dict(before=before,after=after,accepted_fraction=accepted,parameter_changes=changes,
                criterion='Penalized observed teacher likelihood using opposite-fold visual predictions',
                not_target_GT_validation_or_accuracy_guarantee=True)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def train_pair(features,probability,keys,initial,folder,device,registration_sha256,data_sha256):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    folds=group_folds(keys)
    np.save(folder/'folds.npy',folds)
    mean=np.nanmean(probability,axis=1)
    models=[];bootstraps=[]
    for k in range(2):
        model=make_solver(device)
        path=folder/('ordinary_fold'+str(k)+'.pt')
        selected=folds==k
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['registration_sha256']==registration_sha256 and saved['data_sha256']==data_sha256
            assert saved['fit']['adequate_passed'] and saved['trained_fold']==k
            model.load_state_dict(saved['model']);fit=saved['fit']
        else:
            fit=fit_solver(model,features[selected],mean[selected],device,steps=6000,adequate=True)
            saved=dict(model=cpu_state(model),fit=fit,trained_fold=k,seed=0,
                       registration_sha256=registration_sha256,data_sha256=data_sha256,
                       target_sha256=hashlib.sha256(mean[selected].tobytes()).hexdigest())
            save_checkpoint(path,saved)
        models.append(model);bootstraps.append(dict(fit=fit,checkpoint_sha256=file_sha(path)))
        print(json.dumps(dict(status='ADEQUATE_GROUP_DISJOINT_ORDINARY',pair=folder.name,fold=k,
                              labels=int(selected.sum()),steps=fit['steps'],excess_KL=fit['excess_KL'])),flush=True)
    final=folder/'pair_final.pt';saved_path=folder/'pair_closed.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False)
        assert saved['registration_sha256']==registration_sha256 and saved['data_sha256']==data_sha256
        for model,state in zip(models,saved['models']): model.load_state_dict(state)
        assert file_sha(folder/'final_targets.npy')==saved['final_target_sha256']
        atomic_json(folder/'pair_complete.json',dict(status='PAIR_AND_TARGETS_COMPLETE_NO_GT',
                    target_sha256=saved['final_target_sha256'],pair_checkpoint_sha256=file_sha(final),
                    registration_sha256=registration_sha256,data_sha256=data_sha256,GT_open_attempts=0))
        return models,saved
    observer=SemanticPatterns.from_state(initial)
    rounds=[];completed=-1
    if saved_path.exists():
        saved=torch.load(saved_path,map_location='cpu',weights_only=False)
        assert saved['registration_sha256']==registration_sha256 and saved['data_sha256']==data_sha256
        for model,state in zip(models,saved['models']): model.load_state_dict(state)
        observer=SemanticPatterns.from_state(saved['observer'])
        rounds,completed=saved['rounds'],saved['completed']
    for stage in range(completed+1,4):
        outside=outside_predictions(models,features,folds,device)
        r_update=observer.update(probability,outside)
        target=observer.posterior(probability,outside)
        previous=[copy.deepcopy(model.state_dict()) for model in models]
        fits=[]
        for k,model in enumerate(models):
            selected=folds==k
            fits.append(fit_solver(model,features[selected],target[selected],device,
                                   steps=4000 if stage==3 else 500))
        accepted=accept_pair(models,previous,observer,probability,features,folds,device)
        row=dict(stage=stage,observer_update=r_update,solver_fits=fits,solver_acceptance=accepted,
                 opposite_fold_prior_mean=float(outside.mean()),posterior_mean=float(target.mean()),
                 posterior_not_ground_truth=True)
        rounds.append(row)
        atomic_json(folder/('round'+str(stage)+'.json'),row)
        saved=dict(models=[cpu_state(model) for model in models],observer=observer.state(),
                   rounds=rounds,completed=stage,bootstraps=bootstraps,seed=0,
                   registration_sha256=registration_sha256,data_sha256=data_sha256)
        save_checkpoint(saved_path,saved)
        print(json.dumps(dict(status='GROUP_DISJOINT_VISUAL_TEACHER_ROUND',pair=folder.name,stage=stage,
                              observer_change=r_update['maximum_parameter_change'],
                              solver_changes=accepted['parameter_changes'])),flush=True)
    target=observer.posterior(probability,outside_predictions(models,features,folds,device))
    np.save(folder/'final_targets.npy',target)
    saved['final_target_sha256']=file_sha(folder/'final_targets.npy')
    save_checkpoint(final,saved)
    atomic_json(folder/'pair_complete.json',dict(status='PAIR_AND_TARGETS_COMPLETE_NO_GT',
                target_sha256=saved['final_target_sha256'],pair_checkpoint_sha256=file_sha(final),
                registration_sha256=registration_sha256,data_sha256=data_sha256,GT_open_attempts=0))
    return models,saved
