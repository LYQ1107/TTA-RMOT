"""Object/state conditional learner and complete-input ordinary KD controls."""
import torch
from torch import nn
from torch.nn import functional as F
from flex_uda.active.cross_candidate_gauge import GaugeFixedCrossCandidateHead
from flex_uda.active.cross_candidate_learning import CrossCandidateHead

PAIR_KEYS=('anchors_A','present_A','anchors_B','present_B','first','second')


def complete_context(encoder,anchors,present):
    encoded=encoder(anchors)
    summary=CrossCandidateHead.context(encoded,present)
    order=anchors[...,-2].masked_fill(~present,torch.inf).argsort(dim=-1,stable=True)
    ordered=(encoded*present.unsqueeze(-1)).gather(1,order.unsqueeze(-1).expand_as(encoded))
    mask=present.gather(1,order).to(encoded.dtype)
    return torch.cat((summary,ordered.flatten(1),mask),-1)


class CompletePairKD(nn.Module):
    def __init__(self,dimension,width=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(dimension,width),nn.Tanh())
        self.output=nn.Sequential(nn.Linear(12*width+6,2*width),nn.Tanh(),nn.Linear(2*width,4))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,anchors_A,present_A,anchors_B,present_B,first,second):
        a=torch.cat((complete_context(self.encoder,anchors_A,present_A),self.encoder(first)),-1)
        b=torch.cat((complete_context(self.encoder,anchors_B,present_B),self.encoder(second)),-1)
        direct=self.output(torch.cat((a,b),-1))
        reverse=self.output(torch.cat((b,a),-1))[:,[1,0,2,3]]
        return F.log_softmax((direct+reverse).double()/2,-1)


class CompletePointKD(nn.Module):
    def __init__(self,dimension,width=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(dimension,width),nn.Tanh())
        self.output=nn.Sequential(nn.Linear(6*width+3,2*width),nn.Tanh(),nn.Linear(2*width,1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,point_anchors,point_present,point_current):
        context=complete_context(self.encoder,point_anchors,point_present)
        return self.output(torch.cat((context,self.encoder(point_current)),-1)).squeeze(-1).double()


def make_model(recipe,dimension,width):
    if recipe['family']=='structured':
        return GaugeFixedCrossCandidateHead(dimension,width,
            conditional_estimation=recipe['conditional'],object_reference=recipe['reference'],
            stable_identity=recipe.get('identity',True))
    if recipe['family']=='pair':return CompletePairKD(dimension,width)
    if recipe['family']=='point':return CompletePointKD(dimension,width)
    assert recipe['family']=='coarse'
    result=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,1))
    nn.init.zeros_(result[-1].weight);nn.init.zeros_(result[-1].bias)
    return result


def renormalize(model,recipe,old_mean,old_std,mean,std):
    if recipe['family']=='structured':
        model.renormalize(old_mean,old_std,mean,std);return
    layer=model[0] if recipe['family']=='coarse' else model.encoder[0]
    with torch.no_grad():
        om,os,nm,ns=[torch.as_tensor(v,dtype=layer.weight.dtype,device=layer.weight.device)
                     for v in (old_mean,old_std,mean,std)]
        before=layer.weight.clone();layer.weight.mul_(ns/os);layer.bias.add_(before@((nm-om)/os))


def log_probabilities(model,data,recipe):
    family=recipe['family']
    if family in ('structured','pair'):
        return F.log_softmax(model(**{k:data[k] for k in PAIR_KEYS}).double(),-1)
    if family=='point':z=model(data['point_anchors'],data['point_present'],data['point_current'])
    else:z=model(data['coarse_inputs']).squeeze(-1).double()
    return torch.stack((F.logsigmoid(z),F.logsigmoid(-z)),-1)


def pair_marginal_logs(logp):
    return torch.stack((torch.stack((torch.logsumexp(logp[:,[0,2]],-1),torch.logsumexp(logp[:,[1,3]],-1)),-1),
                        torch.stack((torch.logsumexp(logp[:,[1,2]],-1),torch.logsumexp(logp[:,[0,3]],-1)),-1)),1)
