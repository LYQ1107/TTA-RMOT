"""Complete-input reciprocal predictive learning with external teacher checks.

Temporal agreement is a regularizer, never a true label. Observable visual
and state innovations reduce its strength. Both students retain the complete
original input; the memory student additionally receives dense track moments.
Ordinary independent-dual controls use the identical two architectures/data.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch import nn
from flex_uda.active.factor_evidence import MODES
from flex_uda.active.factor_rsi_learning import Inputs,tensor_data,entropy,uid


class Student(nn.Module):
    def __init__(self,memory=False,width=128,dimension=1101):
        super().__init__();self.memory=memory
        self.net=nn.Sequential(nn.Linear((7 if memory else 5)*dimension+3,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))

    def forward(self,bags,present,current,coarse,memory):
        values=[bags.flatten(1),present.to(current.dtype),current,coarse]
        if self.memory:values.append(memory)
        return self.net(torch.cat(values,-1))


def models(device):
    result=[]
    for memory in (False,True):
        torch.manual_seed(0);result.append(Student(memory=memory).to(device))
    return result


class MemoryInputs(Inputs):
    def __init__(self,video):
        super().__init__(video);v=video;dimension=video.dimension
        self.memory=np.empty((len(v.queries),len(v.tracks),2*dimension),dtype='float32')
        by=[np.flatnonzero(v.columns==i) for i in range(len(v.tracks))]
        for qi in range(len(v.queries)):
            rows=np.arange(len(v.obs));raw=v.dense_rows(np.full(len(rows),qi),rows)
            normalized=((raw-self.mean)/self.std).astype('float32')
            for ti,ids in enumerate(by):
                self.memory[qi,ti]=np.r_[normalized[ids].mean(0),normalized[ids].std(0)]
        left=[];right=[]
        for ids in by:
            order=ids[np.argsort(v.obs[ids,0],kind='stable')]
            left.extend(order[:-1]);right.extend(order[1:])
        self.left,self.right=np.asarray(left,int),np.asarray(right,int)
        assert len(self.left)
        rows=np.arange(len(v.obs));raw=v.dense_rows(np.zeros(len(rows),int),rows)
        a,b=self.left,self.right
        rgb=np.square(raw[b,768:1080]-raw[a,768:1080]).mean(-1)
        state=raw[:,-18:-2].astype('float64');delta=state[b]-state[a]
        scale=np.maximum(np.median(np.abs(delta),axis=0),np.maximum(state.std(0)*.01,1e-6))
        innovation=np.square(delta/scale).mean(-1)
        def median_positive(values):
            positive=values[values>0];return max(float(np.median(positive)),1e-12) if len(positive) else 1.
        gap=(v.obs[b,0]-v.obs[a,0]).astype(float);assert (gap>0).all()
        duration=np.exp(-np.maximum(gap/np.median(gap)-1,0))
        self.visual=np.exp(-rgb/median_positive(rgb))*duration
        self.state_weight=self.visual*np.exp(-innovation/median_positive(innovation))
        from flex_uda.active.acquisition import expression_types
        self.dynamic=np.asarray([bool(set(expression_types(q['query_text']))&{'MOTION','SPATIAL','ROLE'}) for q in v.queries])

    def arrays(self,cases):
        result=super().arrays(cases)
        result['memory']=self.memory[[r['local_query'] for r in cases],[r['local_track'] for r in cases]]
        return result

    def at(self,queries,rows):
        v=self.video
        return self.arrays([dict(local_query=int(q),local_track=int(v.columns[r]),
            frame_id=int(v.obs[r,0]),track_id=int(v.obs[r,1])) for q,r in zip(queries,rows)])

    def edge_data(self,indices):
        queries=np.asarray(indices)//len(self.left);edge=np.asarray(indices)%len(self.left)
        weights=np.column_stack((np.where(self.dynamic[queries],self.state_weight[edge],self.visual[edge]),
            self.visual[edge],self.state_weight[edge])).astype('float32')
        return self.at(queries,self.left[edge]),self.at(queries,self.right[edge]),weights


def folds(paid):
    groups=defaultdict(list)
    for i,r in enumerate(paid):groups[int(r['local_query']),int(r['track_id']),int(r['frame_id'])].append(i)
    order=sorted(groups,key=lambda g:hashlib.sha256(json.dumps(g).encode()).hexdigest())
    whole=Counter(int(r['local_query']) for r in paid if r['factor_mode']=='WHOLE')
    roles=Counter(r['factor_mode'] for r in paid);remaining=set(range(len(paid)));selected=[]
    size=max(2,math.ceil(.05*len(groups)))
    for stage in range(3):
        batch=[]
        for g in order:
            ids=groups[g]
            if not set(ids)<=remaining:continue
            own=Counter(paid[i]['factor_mode'] for i in ids)
            if whole[g[0]]-own['WHOLE']<1:continue
            if any(roles[k]-n<1 for k,n in own.items()):continue
            batch.extend(ids);remaining.difference_update(ids);roles.subtract(own);whole[g[0]]-=own['WHOLE']
            if len({(paid[i]['local_query'],paid[i]['track_id'],paid[i]['frame_id']) for i in batch})>=size:break
        assert len(batch)>=2
        selected.append(sorted(batch))
    return selected


def risk_floor(arrays,roles,target,memory):
    values=[arrays['bags'].reshape(len(roles),-1),arrays['present'],arrays['current'],arrays['coarse']]
    if memory:values.append(arrays['memory'])
    packed=np.ascontiguousarray(np.column_stack((*values,roles)),dtype='float32')
    _,inverse=np.unique(packed.view(np.dtype((np.void,4*packed.shape[1]))).ravel(),return_inverse=True)
    count=np.bincount(inverse);mean=np.bincount(inverse,weights=target)/count
    e=entropy(mean)[inverse]
    return {r:float(e[roles==r].mean()) for r in range(3) if np.any(roles==r)}


def probabilities(pair,arrays,device):
    output=[]
    with torch.no_grad():
        for model in pair:
            parts=[]
            for b in range(0,len(arrays['current']),512):
                parts.append(model(**tensor_data({k:v[b:b+512] for k,v in arrays.items()},device)).sigmoid().cpu().numpy())
            output.append(np.concatenate(parts))
    return np.asarray(output)


def teacher_risks(pair,inputs,paid,device,hard=False):
    arrays=inputs.arrays(paid);roles=np.asarray([MODES.index(r['factor_mode']) for r in paid])
    target=np.asarray([r['yes_probability'] for r in paid],float)
    if hard:target=(target>=.5).astype(float)
    result=[]
    for model in pair:
        floor=risk_floor(arrays,roles,target,model.memory);losses=[]
        with torch.no_grad():
            for b in range(0,len(paid),512):
                z=model(**tensor_data({k:v[b:b+512] for k,v in arrays.items()},device)).double()
                y=torch.as_tensor(target[b:b+512],device=device)
                at=torch.as_tensor(roles[b:b+512,None],device=device)
                losses.append(nn.functional.binary_cross_entropy_with_logits(z.gather(1,at).flatten(),y,reduction='none').cpu().numpy())
        ce=np.concatenate(losses)
        result.append({MODES[r]:dict(CE=float(ce[roles==r].mean()),floor=floor[r],
            excess_KL=float(ce[roles==r].mean()-floor[r]),rows=int(sum(roles==r))) for r in floor})
    return result


def update_experts(weights,predictions,paid):
    """Hedge on freshly held-out TEACHER risks, not GT or task correctness."""
    eps=.001;bound=-math.log(eps);eta=math.sqrt(8*math.log(2)/(3*bound**2))
    roles=np.asarray([MODES.index(r['factor_mode']) for r in paid]);y=np.asarray([r['yes_probability'] for r in paid])
    updated=weights.copy();losses={}
    for r in range(3):
        chosen=roles==r
        if not chosen.any():continue
        p=predictions[:,chosen,r].clip(eps,1-eps);target=y[chosen][None]
        risk=(-target*np.log(p)-(1-target)*np.log1p(-p)).mean(-1)
        values=np.log(weights[:,r])-eta*risk;values-=values.max();updated[:,r]=np.exp(values)/np.exp(values).sum()
        losses[MODES[r]]=risk.tolist()
    return updated,dict(losses=losses,learning_rate=eta,clipped_logloss_bound=bound,
        weights_before=weights.tolist(),weights_after=updated.tolist(),
        bound_applies_to_realized_heldout_teacher_loss_sequence_not_HOTA=True)


def curriculum(pair,inputs,weights,generator,device):
    total=len(inputs.video.queries)*len(inputs.left)
    candidate=generator.choice(total,size=min(8192,total),replace=False)
    a,b,w=inputs.edge_data(candidate);pa=probabilities(pair,a,device);pb=probabilities(pair,b,device)
    value=(w*((pa[0]-pb[1])**2+(pa[1]-pb[0])**2)*np.sqrt(weights[0]*weights[1])).sum(-1)
    distribution=np.full(len(value),1/len(value)) if value.sum()==0 else .8*value/value.sum()+.2/len(value)
    take=generator.choice(len(candidate),size=min(1024,len(candidate)),replace=False,p=distribution)
    chosen=candidate[take]
    left,right,weight=inputs.edge_data(chosen)
    return (left,right,weight),dict(candidate_edges=len(candidate),selected_edges=len(chosen),
        selected_edge_ids=chosen.tolist(),selection_SHA256=hashlib.sha256(chosen.tobytes()).hexdigest(),
        epsilon=.2,mean_selected_visual_state_weight=float(weight.mean()),
        peer_disagreement_is_a_curriculum_not_a_true_label=True)


def train(pair,inputs,paid,*,device,steps,hard=False,mutual=False,edges=None,weights=None,minimum_gate=False):
    start=time.perf_counter();arrays=inputs.arrays(paid);data=tensor_data(arrays,device)
    roles=torch.as_tensor([MODES.index(r['factor_mode']) for r in paid],device=device)
    target=torch.as_tensor([r['yes_probability'] for r in paid],dtype=torch.float32,device=device)
    if hard:target=(target>=.5).float()
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in pair]
    multiplier=1.;done=0;reports=None;history=[]
    if mutual:
        assert edges is not None and weights is not None
        a,b,w=edges;left,right=tensor_data(a,device),tensor_data(b,device)
        edge_weight=torch.as_tensor(w,device=device);expert=torch.as_tensor(weights,dtype=torch.float32,device=device)
    limit=16000 if minimum_gate else steps
    for iteration in range(limit):
        for m,o in zip(pair,optimizers):m.train();o.zero_grad(set_to_none=True)
        for begin in range(0,len(paid),512):
            block={k:v[begin:begin+512] for k,v in data.items()}
            for model in pair:
                z=model(**block).gather(1,roles[begin:begin+512,None]).flatten()
                loss=nn.functional.binary_cross_entropy_with_logits(z,target[begin:begin+512],reduction='sum')/len(paid)
                (multiplier*.5*loss).backward()
        if mutual:
            begin=(iteration*128)%len(edge_weight);ids=(torch.arange(128,device=device)+begin)%len(edge_weight)
            pl=[m(**{k:v[ids] for k,v in left.items()}).sigmoid() for m in pair]
            pr=[m(**{k:v[ids] for k,v in right.items()}).sigmoid() for m in pair]
            penalty=0
            for k in range(2):
                peer=1-k
                delta=(pl[k]-pr[peer].detach())**2+(pr[k]-pl[peer].detach())**2
                penalty=penalty+.5*(edge_weight[ids]*expert[peer]*delta).mean()
            penalty.backward()
        for optimizer in optimizers:optimizer.step()
        done=iteration+1
        if minimum_gate and done>=steps and (done==steps or done%1000==0):
            reports=teacher_risks(pair,inputs,paid,device,hard)
            maximum=max(v['excess_KL'] for r in reports for v in r.values())
            history.append(dict(steps=done,maximum_excess_KL=maximum,teacher_multiplier=multiplier))
            if maximum<=.005:break
            multiplier*=2
    for m in pair:m.eval()
    if reports is None:reports=teacher_risks(pair,inputs,paid,device,hard)
    maximum=max(v['excess_KL'] for r in reports for v in r.values())
    if minimum_gate:assert maximum<=.005,('Preserve teacher fidelity gate; unfinished numerical unit',maximum)
    return dict(steps=done,teacher_rows=len(paid),hard=hard,mutual=mutual,
        teacher_risks=reports,maximum_excess_KL=maximum,teacher_fidelity_gate_passed=maximum<=.005,
        optimization_history=history,wall_seconds=time.perf_counter()-start,seed=0,
        no_manual_GT=True,peer_opinions_not_ground_truth=True)
