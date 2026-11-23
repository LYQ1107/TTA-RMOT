"""Budgeted level-set search on an image-only Gaussian graph field.

The covariance is an acquisition proxy, not a calibrated confidence statement.
Only actually requested teacher answers may change the adaptive posterior.
"""
import numpy as np
from scipy.special import logit
from flex_uda.active.visual_classical import visual_graph


class GraphSearch:
    def __init__(self,prior,features,visible=None,*,adaptive=True):
        self.prior=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.)
        self.answers=np.full(len(prior),np.nan);self.adaptive=adaptive
        if visible is None:weights,_=visual_graph(features)
        else:
            ids=np.flatnonzero(visible);weights=np.zeros((len(prior),len(prior)))
            local,_=visual_graph(np.asarray(features)[ids]);weights[np.ix_(ids,ids)]=local
        self.laplacian=np.diag(weights.sum(1))-weights
        self.static_covariance=np.linalg.inv(self.laplacian+np.eye(len(prior))*.1)
        self.static_mean=self.static_covariance@(.1*self.prior)

    def posterior(self):
        if not self.adaptive:return self.static_mean.copy(),self.static_covariance.copy()
        seen=np.isfinite(self.answers);source=np.full(len(seen),.1);source[seen]=0.
        precision=np.zeros(len(seen));precision[seen]=1+3*np.abs(2*self.answers[seen]-1)
        target=np.zeros(len(seen));target[seen]=logit(np.clip(self.answers[seen],.02,.98))
        covariance=np.linalg.inv(self.laplacian+np.diag(source+precision))
        return covariance@(source*self.prior+precision*target),covariance

    def update(self,index,probability):
        assert not np.isfinite(self.answers[index])
        self.answers[index]=probability
        if not self.adaptive:
            c=self.static_covariance;column=c[:,index].copy();increment=.9
            self.static_covariance=c-increment*np.outer(column,column)/(1+increment*c[index,index])


def select_batch(fields,eligible,size):
    means=[];covariance=[];used=[]
    for field in fields:
        mean,cov=field.posterior();means.append(mean);covariance.append(cov)
        used.append(np.isfinite(field.answers).copy())
    selections=[]
    for _ in range(size):
        scores=np.stack([2*np.sqrt(np.maximum(np.diag(c),0))-np.abs(m) for m,c in zip(means,covariance)])
        allowed=np.zeros_like(scores,dtype=bool);allowed[:,eligible]=True
        for qi in range(len(fields)):allowed[qi]&=~used[qi]
        # One examination per provided query comes first whenever the total
        # budget permits it; then allocate by the common uncertainty score.
        unseen_queries=np.asarray([not seen.any() for seen in used])
        if unseen_queries.any():allowed &= unseen_queries[:,None]
        scores[~allowed]=-np.inf
        assert np.isfinite(scores).any(),'No distinct eligible questions remain'
        qi,ti=np.unravel_index(int(np.argmax(scores)),scores.shape)
        selections.append(dict(local_query=int(qi),local_track=int(ti),acquisition_score=float(scores[qi,ti]),
            posterior_logit=float(means[qi][ti]),posterior_std_proxy=float(np.sqrt(max(covariance[qi][ti,ti],0)))))
        used[qi][ti]=True;c=covariance[qi];column=c[:,ti].copy();increment=.9
        # A minimum-precision variance-only fantasy avoids redundant queries
        # within a GPU batch. No fabricated label is used to fit any method.
        covariance[qi]=c-increment*np.outer(column,column)/(1+increment*c[ti,ti])
    return selections
