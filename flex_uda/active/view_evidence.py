"""Regularised binary DS on requested views, used as bounded evidence.

Repeated judgments of the same view first become one vote. The head target
uses mean per-view log likelihood ratio with a neutral prior: selective-QA
prevalence is not exported as a population absence prior, and correlated
Qwen views do not multiply confidence. Reliability remains an unsupervised
model estimate, not observed accuracy.
"""
import numpy as np
from scipy.special import expit,logit

VIEWS=('CROP','CLIP','FRAME_BOX')


def fit(votes,*,iterations=20,alpha=8.,beta=2.):
    observed=np.isfinite(votes);yes=np.where(observed,votes,0.);no=observed-yes
    assert ((votes[observed]==0)|(votes[observed]==1)).all()
    s=np.full(len(VIEWS),.8);c=s.copy();pi=.5
    for _ in range(iterations):
        l1=np.log(pi)+yes@np.log(s)+no@np.log1p(-s)
        l0=np.log1p(-pi)+yes@np.log1p(-c)+no@np.log(c)
        z=expit(l1-l0)
        s=(z@yes+alpha)/(z@observed+alpha+beta)
        c=((1-z)@no+alpha)/((1-z)@observed+alpha+beta)
        pi=float((z.sum()+1)/(len(z)+2))
    return s,c,pi


def evidence(votes,sensitivity,specificity):
    observed=np.isfinite(votes);yes=np.where(observed,votes,0.);no=observed-yes
    s=np.clip(sensitivity,1e-6,1-1e-6);c=np.clip(specificity,1e-6,1-1e-6)
    log_lr=yes@np.log(s/(1-c))+no@np.log((1-s)/c)
    # The fitted prevalence is used internally by DS, never an absence prior.
    return np.clip(expit(log_lr/np.maximum(observed.sum(1),1)),.02,.98)


def targets(view_probability):
    raw=np.asarray(view_probability,dtype=float);known=np.isfinite(raw)
    assert raw.shape[1]==len(VIEWS) and known.any(1).all()
    votes=np.where(known,raw>.5,np.nan)
    overlap=int((known.sum(1)>1).sum());active_views=int(known.any(0).sum())
    # A single-view marginal cannot identify an annotator confusion matrix.
    # Do not turn its positive frequency into an apparent learned accuracy.
    if active_views<2 or overlap==0:
        s=np.full(3,.8);c=s.copy();pi=.5;mode='FIXED_NO_CROSS_VIEW_OVERLAP'
    else:
        s,c,pi=fit(votes);mode='PRIOR_REGULARIZED_DS_NOT_IDENTIFIED_ACCURACY'
    fixed=evidence(votes,np.full(3,.8),np.full(3,.8));adaptive=evidence(votes,s,c)
    mean=np.nansum(raw,axis=1)/known.sum(1)
    return mean,fixed,adaptive,dict(views=list(VIEWS),sensitivity=s.tolist(),specificity=c.tolist(),
        fitted_selection_prevalence=pi,prevalence_exported_to_head=False,
        view_cells=len(raw),view_counts=known.sum(0).tolist(),overlapping_view_cells=overlap,
        reliability_mode=mode,active_views=active_views,
        reliability_identifiability='Single view falls back to fixed .8; two correlated views or sparse triple overlap do not identify true accuracies.',
        beta_prior=[8.,2.],EM_iterations=20,teacher_binary_vote_threshold='>.5',
        repeated_same_view_collapsed=True,mean_view_log_LR_not_independent_confidence_product=True,
        target_probability_bounds=[.02,.98],unsupervised_reliability_not_measured_accuracy=True)
