"""Candidate active observation design from coverage and identifiability.

Stage one pays for object anchors. Freeze those anchors before constructing
stage two: one-unit-cost actions are new-object anchors or another state of an
already anchored object. The objective is a conditional submodular proxy, not
true information gain, a global two-stage guarantee, or a bound on HOTA.
"""
import numpy as np
from scipy.spatial.distance import cdist

ANCHOR_FRACTION=.75
INFORMATION_WEIGHT=.25
INFORMATION_RIDGE=.01


def similarity(features):
    x=np.asarray(features,float);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    distance=np.maximum(cdist(x,x,'sqeuclidean'),0.)
    positive=distance[distance>1e-12];scale=float(np.median(positive)) if len(positive) else 1.
    values=np.exp(-distance/max(scale,1e-12));np.fill_diagonal(values,1.)
    return values


def greedy_anchors(kernel,weights,count,prior):
    kernel=np.asarray(kernel,float);weights=np.asarray(weights,float);weights=weights/weights.sum()
    prior=np.asarray(prior,float);covered=np.zeros(len(kernel));selected=[];trace=[]
    for _ in range(min(count,len(kernel))):
        options=[t for t in range(len(kernel)) if t not in selected]
        gains={t:float(weights@np.maximum(kernel[:,t]-covered,0)) for t in options}
        chosen=min(options,key=lambda t:(-gains[t],-prior[t],t))
        selected.append(chosen);covered=np.maximum(covered,kernel[:,chosen]);trace.append(gains[chosen])
    return selected,covered,trace


class ConditionalObjective:
    """Fixed nonnegative similarities/designs and paid anchor set required.

    F(S)=(1-eta)*weighted facility coverage(S union anchors)
          +eta*log(1+sum_t max_{a in S on t} contrast_a^2/ridge)/normalizer.

    Maximum per object prevents re-counting several contrasts with a shared
    anchor as independent information. It remains a geometric excitation
    proxy, not Fisher information for the actual Huber/teacher model.
    """
    def __init__(self,kernel,weights,initial_coverage,actions,budget,eta=INFORMATION_WEIGHT,ridge=INFORMATION_RIDGE):
        self.kernel=np.asarray(kernel,float);self.weights=np.array(weights,dtype=float,copy=True);self.weights/=self.weights.sum()
        self.initial=np.asarray(initial_coverage,float);self.actions=actions;self.eta=eta;self.ridge=ridge
        assert 0<=eta<=1 and ridge>0 and np.min(self.kernel)>=0 and np.min(self.weights)>=0
        assert len(self.initial)==len(self.kernel) and all(a['contrast_squared']>=0 for a in actions)
        self.normalizer=float(np.log1p(max(budget,1)*.25/ridge))

    def state(self,selected):
        covered=self.initial.copy();excitation=np.zeros(len(self.kernel))
        for index in selected:
            action=self.actions[index];t=action['track_index'];covered=np.maximum(covered,self.kernel[:,t])
            excitation[t]=max(excitation[t],action['contrast_squared'])
        return covered,excitation

    def value(self,selected):
        covered,excitation=self.state(selected)
        return float((1-self.eta)*(self.weights@covered)+self.eta*np.log1p(excitation.sum()/self.ridge)/self.normalizer)

    def select(self,budget,prior):
        selected=[];covered=self.initial.copy();excitation=np.zeros(len(self.kernel));trace=[]
        assert budget<=len(self.actions)
        for _ in range(budget):
            current=float(excitation.sum());options=[]
            for index,action in enumerate(self.actions):
                if index in selected:continue
                t=action['track_index'];new=max(excitation[t],action['contrast_squared'])-excitation[t]
                coverage=float(self.weights@np.maximum(self.kernel[:,t]-covered,0))
                information=float(np.log1p(new/(self.ridge+current))/self.normalizer)
                gain=(1-self.eta)*coverage+self.eta*information
                options.append((-gain,-float(prior[t]),int(action['frame']),t,index))
            neg,_,_,_,chosen=min(options);selected.append(chosen);action=self.actions[chosen];t=action['track_index']
            covered=np.maximum(covered,self.kernel[:,t]);excitation[t]=max(excitation[t],action['contrast_squared']);trace.append(-neg)
        return selected,trace


def plan(features,weights,prior,samples,anchor_frames,positions,budget,direction):
    """Return a geometry-only contingent plan; no teacher answer is an input.

    samples[t] lists unique candidate frames, including paid anchor_frames[t].
    positions[t] maps each sampled frame to normalized box-centre x. The
    geometry-only follow-up plan can be frozen before any teacher request.
    """
    n=len(features);assert n==len(weights)==len(prior)==len(samples)==len(anchor_frames)==len(positions)
    assert budget>=0 and all(int(a) in set(map(int,s)) for a,s in zip(anchor_frames,samples))
    assert all(0<=float(value)<=1 for mapping in positions for value in mapping.values())
    kernel=similarity(features);first_count=min(n,int(np.ceil(ANCHOR_FRACTION*budget)))
    first,covered,first_gain=greedy_anchors(kernel,weights,first_count,prior)
    actions=[]
    for t in range(n):
        if t not in first:
            actions.append(dict(track_index=t,frame=int(anchor_frames[t]),kind='NEW_OBJECT',contrast_squared=0.))
        else:
            anchor=int(anchor_frames[t]);reference=float(positions[t][anchor])
            for frame in sorted(set(map(int,samples[t]))-{anchor}):
                displacement=(float(positions[t][frame])-reference) if direction else 0.
                actions.append(dict(track_index=t,frame=frame,kind='STATE_CONTRAST',contrast_squared=.25*displacement**2,
                    paid_anchor_frame=anchor))
    remaining=budget-len(first)
    assert remaining<=len(actions),dict(budget=budget,paid_anchors=len(first),available_followups=len(actions))
    objective=ConditionalObjective(kernel,weights,covered,actions,remaining)
    chosen,second_gain=objective.select(remaining,prior)
    selected=[dict(track_index=t,frame=int(anchor_frames[t]),kind='PAID_ANCHOR',contrast_squared=0.) for t in first]+[actions[i] for i in chosen]
    assert len(selected)==budget and len({(a['track_index'],a['frame']) for a in selected})==budget
    return selected,dict(paid_anchors=len(first),followups=remaining,
        state_contrasts=sum(a['kind']=='STATE_CONTRAST' for a in selected),
        distinct_objects=len({a['track_index'] for a in selected}),stage1_coverage_marginal=first_gain,
        stage2_proxy_marginal=second_gain,stage2_proxy_initial=objective.value([]),stage2_proxy_final=objective.value(chosen),
        conditional_equal_cost_proxy_only=True,teacher_or_GT_answers_used=False,
        shared_anchor_contrasts_not_summed_as_independent_information=True)
