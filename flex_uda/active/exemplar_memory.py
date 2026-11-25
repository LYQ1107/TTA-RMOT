"""Gaussian visual exemplar memory with an explicit epistemic variance.

Only frozen image/query descriptors, source scores and teacher probabilities
are inputs. Bandwidth is obtained from descriptor geometry, without labels.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.special import expit,logit


def normalized_features(x):
    x=np.asarray(x,dtype=np.float64)
    x=x-x.mean(1,keepdims=True)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    return x


class ExemplarMemory:
    def fit(self,features,source_logits,teacher_probability):
        self.x=normalized_features(features)
        distance=np.maximum(cdist(self.x,self.x,'sqeuclidean'),0.)
        n=len(self.x);rank=min(10,n-1)
        np.fill_diagonal(distance,np.inf)
        self.bandwidth=max(float(np.median(np.partition(distance,rank-1,axis=1)[:,rank-1])),1e-6) if rank else 1.
        np.fill_diagonal(distance,0.)
        self.kernel=np.exp(-distance/self.bandwidth)
        # A finite observation noise prevents an isolated teacher answer from
        # becoming an infinitely confident constraint on its neighbours.
        self.factor=cho_factor(self.kernel+np.eye(n)*.1,lower=True,check_finite=False)
        target=np.clip(logit(np.clip(teacher_probability,1e-6,1-1e-6)),-4.,4.)-np.asarray(source_logits)
        self.alpha=cho_solve(self.factor,target,check_finite=False)
        return dict(examples=n,bandwidth=self.bandwidth,regularizer=.1,bandwidth_uses_labels=False)

    def predict(self,features,source_logits):
        x=normalized_features(features)
        distance=np.maximum(cdist(x,self.x,'sqeuclidean'),0.)
        k=np.exp(-distance/self.bandwidth)
        value=np.asarray(source_logits)+k@self.alpha
        variance=1-np.einsum('ij,ji->i',k,cho_solve(self.factor,k.T,check_finite=False))
        return expit(value),np.clip(variance,0,1)
