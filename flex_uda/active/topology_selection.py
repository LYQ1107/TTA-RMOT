"""Teacher-only predictive choice of local versus scene-wide reference.

Validation uses neutral unaries, never the teacher-trained neural head. All
labels for a held TrackID are excluded across queries. The scoring contract is
provided by the task: balanced identity evidence for PRESUPPOSED, Bernoulli
prediction for CHECKLIST. This is not the older Gaussian-evidence BMA model.
"""
import hashlib
import numpy as np
from flex_uda.active.grounded_graph import query_geometry
from flex_uda.active.visual_classical import visual_graph,refine_distilled_prior


def graph_matrices(pcd,colours,texts):
    parts,geometry=query_geometry(pcd,colours,texts)
    local=[];global_=[];valid=[];supported=[]
    for q,mask in enumerate(parts):
        visible=np.all(colours[:,mask>0].sum(2)>0,axis=1) if mask.any() else np.zeros(len(colours),bool)
        ids=np.flatnonzero(visible);small,_=visual_graph(geometry[q,ids])
        dense=np.ones_like(small)-np.eye(len(ids))
        if dense.sum():dense*=small.sum()/dense.sum()
        pair=[]
        for block in (small,dense):
            matrix=np.zeros((len(colours),len(colours)));matrix[np.ix_(ids,ids)]=block;pair.append(matrix)
        local.append(pair[0]);global_.append(pair[1]);valid.append(visible);supported.append(bool(mask.any()))
    return geometry,np.asarray(local),np.asarray(global_),np.asarray(valid),np.asarray(supported)


def apply_graph(unary,teacher,geometry,weights,supported):
    result=np.array(unary,copy=True)
    for q in np.flatnonzero(supported):
        result[q]=refine_distilled_prior(unary[q],geometry[q],teacher[q],source_weight=.1,weights_override=weights[q])
    return result


def teacher_crossfit_choice(pcd,colours,texts,teacher,tracks,video,protocol):
    assert protocol in ('PRESUPPOSED','CHECKLIST')
    geometry,local,global_,valid,supported=graph_matrices(pcd,colours,texts)
    folds=np.asarray([int(hashlib.sha256((video+':'+str(int(t))+':semantic-fold').encode()).hexdigest()[:8],16)%5 for t in tracks])
    known=np.isfinite(teacher)&valid&supported[:,None]
    predictions=np.full((2,)+teacher.shape,.5,dtype=float)
    neutral=np.full(teacher.shape,.5,dtype=float)
    for fold in sorted(set(folds)):
        held=folds==fold;training=np.asarray(teacher).copy();training[:,held]=np.nan
        # No fitted KD probability is used anywhere in validation. A held
        # answer cannot re-enter through an overfitted neural unary.
        for model,weights in enumerate((local,global_)):
            probability=apply_graph(neutral,training,geometry,weights,supported)
            predictions[model][:,held]=probability[:,held]
    losses={name:[] for name in ('BCE','BALANCED')};queries=0;observations=0
    for q in np.flatnonzero(supported):
        selected=known[q]
        if not selected.any():continue
        y=np.clip(teacher[q,selected],.02,.98);p=np.clip(predictions[:,q][:,selected],1e-6,1-1e-6)
        assert p.shape==(2,len(y))
        positive=-(np.log(p)*y[None,:]);negative=-(np.log1p(-p)*(1-y)[None,:])
        losses['BCE'].append((positive+negative).mean(1))
        losses['BALANCED'].append(.5*(positive.sum(1)/y.sum()+negative.sum(1)/(1-y).sum()))
        queries+=1;observations+=len(y)
    average={name:np.mean(values,axis=0) if values else np.asarray([0.,0.]) for name,values in losses.items()}
    choices={name:int(value[1]<value[0]) for name,value in average.items()}
    key='BALANCED' if protocol=='PRESUPPOSED' else 'BCE';selected=choices[key]
    receipt=dict(protocol=protocol,selected_model=('LOCAL','GLOBAL')[selected],BCE_selected_model=('LOCAL','GLOBAL')[choices['BCE']],
        BALANCED_selected_model=('LOCAL','GLOBAL')[choices['BALANCED']],validation_losses={k:v.tolist() for k,v in average.items()},
        teacher_validation_queries=queries,held_teacher_observations=observations,TrackID_folds=5,
        fitted_KD_unaries_used_in_validation=False,validation_unary=.5,all_held_track_answers_excluded_across_queries=True,
        no_manual_target_GT=True,no_future_teacher=True,tie_rule='LOCAL',teacher_predictive_consistency_not_GT_accuracy=True)
    return geometry,local,global_,supported,selected,choices,receipt
