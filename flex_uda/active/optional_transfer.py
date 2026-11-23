"""Teacher-predictive graph selection includes an identity operation.

Held TrackIDs are excluded from all current queries. The fitted KD head never
enters validation. One paired-query SE is a sensitivity regularizer, not a
calibrated confidence interval, because queries share tracks and evidence.
"""
import hashlib
import numpy as np
from flex_uda.active.topology_selection import graph_matrices,apply_graph


def optional_graphs(unary,pcd,colours,texts,teacher,tracks,video,protocol):
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
    identity_risk=float(np.log(2.));key='BALANCED' if protocol=='PRESUPPOSED' else 'BCE'
    average={name:np.mean(values,axis=0) if values else np.full(2,identity_risk) for name,values in losses.items()}
    forced=int(average[key][1]<average[key][0]);pairs=np.asarray(losses[key]).reshape(-1,2)
    gain=identity_risk-pairs[:,forced];n=len(gain)
    mean=float(gain.mean()) if n else 0.;se=float(gain.std(ddof=1)/np.sqrt(n)) if n>1 else float('inf')
    # Only suppress machine-level cancellation, not a target-tuned margin.
    epsilon=float(32*np.finfo(float).eps)
    primary=forced+1 if n>1 and mean>max(se,epsilon) else 0
    strict=forced+1 if mean>epsilon else 0
    full_scores=[np.array(unary,copy=True),apply_graph(unary,teacher,geometry,local,supported),apply_graph(unary,teacher,geometry,global_,supported)]
    names=('IDENTITY','LOCAL','GLOBAL')
    scores=dict(IDENTITY=full_scores[0],FULL=full_scores[1],GLOBAL=full_scores[2],
        OPTIONAL=full_scores[primary],STRICT=full_scores[strict],FORCED=full_scores[forced+1])
    anchor_only=apply_graph(unary,teacher,geometry,np.zeros_like(local),supported)
    scores['ANCHOR_OPTIONAL']=anchor_only if primary==0 else full_scores[primary]
    receipt=dict(protocol=protocol,selected_model=names[primary],strict_selected_model=names[strict],forced_selected_model=names[forced+1],
        validation_losses={k:v.tolist() for k,v in average.items()},identity_validation_risk=identity_risk,
        paired_query_gain_over_identity=mean,paired_query_gain_SE=se if np.isfinite(se) else None,
        teacher_validation_queries=n,held_teacher_observations=observations,TrackID_folds=5,
        primary_rule='BestgraphmustbeatidentitybyonepairedquerySE;elsesharedKDunchanged',
        validation_unary=.5,fitted_KD_unaries_used_in_validation=False,current_held_TrackID_answers_excluded_across_queries=True,
        identity_deployment='ExactKDidentitynotzeroedgegraphwithrawteacheranchors',machine_cancellation_epsilon=epsilon,
        raw_teacher_anchor_fallback_control_included=True,
        no_manual_target_GT=True,no_future_teacher=True,SE_is_sensitivity_regularizer_not_calibrated_confidence=True,
        query_and_crossfit_dependencies_not_removed_by_SE=True,teacher_predictive_consistency_not_GT_accuracy=True)
    return scores,receipt
