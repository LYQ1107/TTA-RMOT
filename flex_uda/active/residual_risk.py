"""Predictable control variates for a declared teacher-response action pool.

Classical augmented importance sampling, not a new estimator theorem. The
scope is fixed-model teacher risk under recorded with-replacement draws.
Neither semantic correctness nor learned-model generalization follows.
"""
import numpy as np


def coverage_distribution(priority, floor=.2):
    priority=np.asarray(priority,dtype='float64')
    assert priority.ndim==1 and len(priority) and np.isfinite(priority).all() and (priority>=0).all()
    assert 0<floor<=1
    adaptive=priority/priority.sum() if priority.sum()>0 else np.full(len(priority),1/len(priority))
    result=floor/len(priority)+(1-floor)*adaptive
    assert np.isclose(result.sum(),1.) and (result>=floor/len(priority)-1e-15).all()
    return result


class ResidualRisk:
    def __init__(self,query):
        self.query=np.asarray(query,dtype=int)
        self.groups={int(q):np.flatnonzero(self.query==q) for q in np.unique(self.query)}
        self.proxy_sum=np.zeros(len(query));self.correction_sum=np.zeros(len(query))
        self.draws={q:0 for q in self.groups}

    def record(self,query,proxy_before,selected,teacher_probability,proposal_probability):
        group=self.groups[int(query)];m=np.asarray(proxy_before,dtype='float64')
        assert m.shape==(len(group),) and np.isfinite(m).all() and ((m>=0)&(m<=1)).all()
        assert selected in group and 0<=teacher_probability<=1 and 0<proposal_probability<=1
        local=int(np.flatnonzero(group==selected)[0])
        self.proxy_sum[group]+=m
        self.correction_sum[selected]+=(teacher_probability-m[local])/proposal_probability
        self.draws[int(query)]+=1

    def coefficients(self):
        assert min(self.draws.values())>0
        denom=np.asarray([self.draws[int(q)] for q in self.query],dtype='float64')
        # These are signed risk coefficients, NOT probability pseudo-labels.
        # Clipping them to[0,1] destroys the cancellation identity.
        return (self.proxy_sum+self.correction_sum)/denom

    def population_weights(self):
        return np.asarray([1/(len(self.groups)*len(self.groups[int(q)])) for q in self.query])


def numpy_risk(logits,coefficient,weights):
    z=np.asarray(logits,dtype='float64');c=np.asarray(coefficient,dtype='float64');w=np.asarray(weights,dtype='float64')
    assert z.shape==c.shape==w.shape and np.isfinite(z).all() and np.isfinite(c).all()
    return float(np.sum(w*(np.logaddexp(0,z)-c*z)))


def torch_risk(logits,coefficient,weights):
    import torch
    return torch.sum(weights*(torch.nn.functional.softplus(logits)-coefficient*logits))


def bounded_logits(raw,cap=8.):
    return cap*(raw/cap).tanh()
