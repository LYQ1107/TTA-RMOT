"""Conservative teacher predictive selection with fixed one-SE sensitivity.

The SE summarizes variation of query losses, not a calibrated confidence
interval: queries and cross-fit predictions share visual tracks. All geometry
and held-TrackID fitting exactly follow the frozen strict-minimum selector.
"""
import hashlib
import numpy as np
from flex_uda.active.topology_selection import graph_matrices,apply_graph


def teacher_stability_choice(pcd,colours,texts,teacher,tracks,video,protocol):
    assert protocol in ('PRESUPPOSED','CHECKLIST')
    geometry,local,global_,valid,supported=graph_matrices(pcd,colours,texts)
    folds=np.asarray([int(hashlib.sha256((video+':'+str(int(t))+':semantic-fold').encode()).hexdigest()[:8],16)%5 for t in tracks])
    known=np.isfinite(teacher)&valid&supported[:,None]
    predictions=np.full((2,)+teacher.shape,.5,dtype=float);neutral=np.full(teacher.shape,.5,dtype=float)
    for fold in sorted(set(folds)):
        held=folds==fold;training=np.asarray(teacher).copy();training[:,held]=np.nan
        for model,weights in enumerate((local,global_)):
            probability=apply_graph(neutral,training,geometry,weights,supported)
            predictions[model][:,held]=probability[:,held]
    losses={name:[] for name in ('BCE','BALANCED')};observations=0
    for q in np.flatnonzero(supported):
        selected=known[q]
        if not selected.any():continue
        y=np.clip(teacher[q,selected],.02,.98);p=np.clip(predictions[:,q][:,selected],1e-6,1-1e-6)
        positive=-(np.log(p)*y[None,:]);negative=-(np.log1p(-p)*(1-y)[None,:])
        losses['BCE'].append((positive+negative).mean(1))
        losses['BALANCED'].append(.5*(positive.sum(1)/y.sum()+negative.sum(1)/(1-y).sum()))
        observations+=len(y)
    average={name:np.mean(values,axis=0) if values else np.asarray([0.,0.]) for name,values in losses.items()}
    choices={name:int(value[1]<value[0]) for name,value in average.items()}
    key='BALANCED' if protocol=='PRESUPPOSED' else 'BCE';pairs=np.asarray(losses[key]).reshape(-1,2)
    delta=pairs[:,0]-pairs[:,1];n=len(delta);mean=float(delta.mean()) if n else 0.
    se=float(delta.std(ddof=1)/np.sqrt(n)) if n>1 else float('inf')
    choices.update(ONE_SE=int(n>1 and mean>se),TWO_SE=int(n>1 and mean>2*se),
        MEDIAN=int(n>1 and np.median(delta)>0),DELETE_ONE=int(n>1 and np.min((delta.sum()-delta)/(n-1))>0))
    selected=choices['ONE_SE']
    receipt=dict(protocol=protocol,selected_model=('LOCAL','GLOBAL')[selected],
        BCE_selected_model=('LOCAL','GLOBAL')[choices['BCE']],BALANCED_selected_model=('LOCAL','GLOBAL')[choices['BALANCED']],
        control_selected_models={k:('LOCAL','GLOBAL')[choices[k]] for k in ('ONE_SE','TWO_SE','MEDIAN','DELETE_ONE')},
        validation_losses={k:v.tolist() for k,v in average.items()},paired_query_risk_difference=mean,
        paired_query_risk_SE=se if np.isfinite(se) else None,teacher_validation_queries=n,held_teacher_observations=observations,
        TrackID_folds=5,fitted_KD_unaries_used_in_validation=False,validation_unary=.5,
        all_held_track_answers_excluded_across_queries=True,no_manual_target_GT=True,no_future_teacher=True,
        tie_rule='LOCAL',primary_rule='ONE_SE',SE_is_sensitivity_regularizer_not_calibrated_confidence=True,
        query_and_crossfit_dependencies_not_removed_by_SE=True,teacher_predictive_consistency_not_GT_accuracy=True)
    return geometry,local,global_,supported,selected,choices,receipt
