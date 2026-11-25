"""Beta peer-consistency audit of noisy teacher answers, without target labels."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.special import betaincc


def audit_weights(features,answer,visible=None,*,use_cohorts=True):
    answer=np.asarray(answer,dtype=float);n=len(answer);seen=np.isfinite(answer)
    active=np.ones(n,dtype=bool) if visible is None else np.asarray(visible,dtype=bool)
    x=np.array(features,dtype=float,copy=True);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    distance=np.maximum(cdist(x,x,'sqeuclidean'),0.)
    selected=np.flatnonzero(active);adjacency=np.zeros((n,n),dtype=bool)
    if use_cohorts and len(selected)>1:
        d=distance[np.ix_(selected,selected)].copy();np.fill_diagonal(d,np.inf)
        k=min(2,len(selected)-1);nearest=np.argsort(d,axis=1,kind='stable')[:,:k]
        local=np.zeros_like(d,dtype=bool);np.put_along_axis(local,nearest,True,axis=1)
        adjacency[np.ix_(selected,selected)]=local&local.T
    elif not use_cohorts:adjacency=active[:,None]&active[None,:]
    _,labels=connected_components(adjacency,directed=False)
    factors=np.ones(n);rows=[]
    for i in np.flatnonzero(seen&active):
        peers=np.flatnonzero(seen&active&(labels==labels[i])&(np.arange(n)!=i))
        # Three distinct candidates are the minimum useful majority. A lone
        # rare candidate cannot be overruled by unrelated majority evidence.
        if len(peers)<2:continue
        alpha=1.+answer[peers].sum();beta=1.+len(peers)-answer[peers].sum()
        positive=float(betaincc(alpha,beta,.5))
        agreement=positive if answer[i]>=.5 else 1-positive
        factors[i]=min(1.,agreement/max(1-agreement,1e-12))
        rows.append(dict(track_index=int(i),peer_track_indices=peers.tolist(),peer_positive_probability=positive,
            agreement=agreement,relative_teacher_precision=float(factors[i])))
    return factors,dict(mutual_neighbours=2,use_visual_cohorts=use_cohorts,minimum_other_distinct_peers=2,
        downweighted=int((factors[seen]<1-1e-8).sum()),supported=len(rows),rows=rows,
        beta_prior=[1.,1.],no_GT_used=True)
