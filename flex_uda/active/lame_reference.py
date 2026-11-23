"""CPU binary-query adapter for the published LAME baseline, not our method.

Based on Boudiaf, Mueller, Ben Ayed, Bertinetto, CVPR2022 and fiveai/LAME
commit d2e5f63090bc1c8129bf7cbd781029a5955e1a67, src/adaptation/lame.py.
Original/adapted code: CC BY-NC-SA4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/
Source: https://github.com/fiveai/LAME/tree/d2e5f63090bc1c8129bf7cbd781029a5955e1a67
Modified: NumPy/SciPy float64 CPU inference, explicit self exclusion and stable
Track-ID-order ties. Same k=5, lambda=1,100steps,energy tolerance1e-8; no tuning.
This adapts LAME to query-track binary batches; not the original classification
benchmark or a claim that its convergence theorem holds for arbitrary kernels.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax


def kernel(features,neighbors=5):
    x=np.asarray(features,dtype=np.float64)
    assert x.ndim==2 and np.isfinite(x).all()
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    n=len(x);weights=np.zeros((n,n),dtype=np.float64)
    if n<2:return weights
    distance=cdist(x,x);np.fill_diagonal(distance,np.inf)
    ids=np.argsort(distance,axis=1,kind='stable')[:,:min(neighbors,n-1)]
    np.put_along_axis(weights,ids,1.,axis=1)
    return weights


def optimize(unary,weights,max_steps=100):
    unary=np.asarray(unary,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    assert unary.ndim==2 and weights.shape==(len(unary),len(unary))
    y=softmax(-unary,axis=1);old=float('inf');energies=[];converged=False
    for step in range(max_steps):
        pairwise=weights@y;y=softmax(-unary+pairwise,axis=1)
        energy=float(np.sum(unary*y-pairwise*y+y*np.log(np.maximum(y,1e-20))))
        energies.append(energy)
        if step>1 and abs(energy-old)<=1e-8*abs(old):converged=True;break
        old=energy
    assert np.isfinite(y).all() and np.max(np.abs(y.sum(1)-1))<1e-12
    return y,dict(iterations=len(energies),energy_stopping_met=converged,
        first_energy=energies[0],last_energy=energies[-1],lambda_value=1.,max_steps=max_steps,
        no_gradient_or_network_updates=True)


def binary(probability,features):
    p=np.asarray(probability,dtype=np.float64)
    assert p.ndim==1 and len(features)==len(p) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    weights=kernel(features)
    if not len(p):return p.copy(),dict(iterations=0,energy_stopping_met=True)
    unary=-np.log(np.stack((1-p,p),axis=1)+1e-10)
    y,receipt=optimize(unary,weights)
    return y[:,1],dict(receipt,neighbors=min(5,max(0,len(p)-1)),samples=len(p),
        self_edges=0,affinity='published directed kNN5',ties='stable Track-ID order')
