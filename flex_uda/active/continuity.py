"""Conservative tracklet stitching using motion, appearance and semantic cues.

Candidate boxes and semantic selections never change. Only non-overlapping
source tracks may share an output ID; all costs use unlabeled observations.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy


def unit(x):
    x=np.asarray(x,dtype=float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def semantic_distance(a,b):
    a=np.clip(a,1e-6,1-1e-6);b=np.clip(b,1e-6,1-1e-6);m=.5*(a+b)
    divergence=.5*(xlogy(a,a/m)+xlogy(1-a,(1-a)/(1-m))+
        xlogy(b,b/m)+xlogy(1-b,(1-b)/(1-m)))
    weights=np.maximum(a,b)
    return float(np.sum(weights*divergence)/max(weights.sum(),1e-8)/np.log(2))


def stitch(nodes,*,use_semantics=True,max_gap=30):
    n=len(nodes)
    own=np.asarray([np.square(a['start_visual']-a['end_visual']).sum() for a in nodes])
    scale=max(float(np.quantile(own,.9)),1e-4) if n else 1.
    benefit=np.full((n,n),-1e6);proposals=[]
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            gap=b['first_frame']-a['last_frame']
            if not 0<gap<=max_gap:continue
            delta=b['start_time']-a['end_time']
            predicted=a['end_center']+a['velocity']*delta
            height=max(.5*(a['end_height']+b['start_height']),1.)
            motion=float(np.square((predicted-b['start_center'])/height).sum())
            appearance=float(np.square(a['end_visual']-b['start_visual']).sum())/scale
            if motion>1 or appearance>1:continue
            semantic=semantic_distance(a['end_semantic'],b['start_semantic'])
            value=1-.5*(motion+appearance)-(semantic if use_semantics else 0.)
            benefit[i,j]=value
            proposals.append(dict(source=int(a['track']),destination=int(b['track']),motion=motion,
                appearance=appearance,semantic=semantic,benefit=value))
    # A dummy successor has zero benefit. Bipartite assignment gives one
    # predecessor/successor per tracklet; strict time order prevents cycles.
    cost=np.column_stack([-benefit,np.zeros((n,n))])
    left,right=linear_sum_assignment(cost);links=[(int(i),int(j)) for i,j in zip(left,right) if j<n and benefit[i,j]>0]
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in links:
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    output={int(a['track']):min(int(nodes[j]['track']) for j in range(n) if find(j)==find(i)) for i,a in enumerate(nodes)}
    return output,dict(appearance_scale=scale,appearance_scale_uses_GT=False,max_gap=max_gap,
        semantic_cue_used=use_semantics,proposals=proposals,
        links=[dict(source=int(nodes[i]['track']),destination=int(nodes[j]['track']),benefit=float(benefit[i,j])) for i,j in links])
