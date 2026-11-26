"""Conditional estimation with separate count and relative-state parameters.

This is a candidate RMOT learning mechanism. Conditionalization is a classic
nuisance-removal idea, not a claim of a new theorem or true identity recovery.
The full likelihood ablation has identical parameters and forward values.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalJointObservationHead(nn.Module):
    def __init__(self,dimension,width=64,*,conditional_estimation=True,object_reference=True):
        super().__init__()
        self.conditional_estimation=conditional_estimation
        self.object_reference=object_reference
        # Independent parameter sets are necessary: stopping a local score
        # alone does not isolate it if a count loss updates its shared encoder.
        self.count_encoder=nn.Sequential(nn.Linear(dimension,width),nn.Tanh())
        self.state_encoder=nn.Sequential(nn.Linear(dimension,width),nn.Tanh())
        self.object=nn.Sequential(nn.Linear(2*width,width),nn.Tanh(),nn.Linear(width,1))
        self.local=nn.Sequential(nn.Linear(3*width,width),nn.Tanh(),nn.Linear(width,1))
        self.interaction=nn.Sequential(nn.Linear(4*width,width),nn.Tanh(),nn.Linear(width,1))
        for module in (self.object,self.local,self.interaction):
            nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)

    @staticmethod
    def context(encoded,present):
        assert torch.all(present.any(-1))
        weight=present.unsqueeze(-1).to(encoded.dtype)
        mean=(encoded*weight).sum(-2)/weight.sum(-2)
        maximum=encoded.masked_fill(~present.unsqueeze(-1),-torch.inf).max(-2).values
        return torch.cat((mean,maximum),-1)

    def local_scores(self,anchors,present,first,second):
        a,b=self.state_encoder(first),self.state_encoder(second)
        if self.object_reference:
            encoded=self.state_encoder(anchors);context=self.context(encoded,present)
            ga=self.local(torch.cat((context,a),-1)).squeeze(-1)
            gb=self.local(torch.cat((context,b),-1)).squeeze(-1)
            repeated=context.unsqueeze(-2).expand(*encoded.shape[:-1],context.shape[-1])
            reference=self.local(torch.cat((repeated,encoded),-1)).squeeze(-1)
            offset=(reference*present).sum(-1)/present.sum(-1)
            ga,gb=ga-offset,gb-offset
        else:
            # Same parameters and widths, but a local state cannot read other
            # observations. The count branch still gets the identical context.
            ga=self.local(torch.cat((a,a,a),-1)).squeeze(-1)
            gb=self.local(torch.cat((b,b,b),-1)).squeeze(-1)
        return ga,gb

    def forward(self,anchors,present,first,second):
        ga,gb=self.local_scores(anchors,present,first,second)
        encoded=self.count_encoder(anchors);context=self.context(encoded,present)
        a,b=self.count_encoder(first),self.count_encoder(second)
        base=self.object(context).squeeze(-1)
        interaction=self.interaction(torch.cat((context,torch.abs(a-b),a*b),-1)).squeeze(-1)
        ca,cb=(ga.detach(),gb.detach()) if self.conditional_estimation else (ga,gb)
        # N=0,1,2 probabilities. Count gradients cannot enter local parameters
        # in the conditional-estimation variant; forward values still equal
        # the unrestricted-gradient version at the same parameter values.
        count=F.log_softmax(torch.stack((torch.zeros_like(base),
            base+torch.logaddexp(ca,cb),2*base+ca+cb+interaction),-1),-1)
        relative=F.log_softmax(torch.stack((ga,gb),-1),-1)
        return torch.stack((count[...,1]+relative[...,0],count[...,1]+relative[...,1],
            count[...,2],count[...,0]),-1)

    def renormalize(self,old_mean,old_std,new_mean,new_std):
        # Both independent encoders preserve their original function under
        # the same positive affine prefix-normalization coordinate change.
        for encoder in (self.count_encoder,self.state_encoder):
            layer=encoder[0]
            with torch.no_grad():
                om,os,nm,ns=[torch.as_tensor(v,device=layer.weight.device,dtype=layer.weight.dtype)
                    for v in (old_mean,old_std,new_mean,new_std)]
                assert torch.all(os > 0) and torch.all(ns > 0)
                before=layer.weight.clone();layer.weight.mul_(ns/os)
                layer.bias.add_(before@((nm-om)/os))
