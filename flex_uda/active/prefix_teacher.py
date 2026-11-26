"""Teacher-only prefix adaptation; row normalization has no future statistics."""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PrefixTeacherHead:
    def __init__(self,device,seed=0,feature_dim=768):
        self.device=torch.device(device);torch.manual_seed(seed)
        self.feature_dim=feature_dim
        self.model=nn.Sequential(nn.Linear(feature_dim,64),nn.GELU(),nn.Linear(64,1)).to(self.device)
        nn.init.zeros_(self.model[-1].weight);nn.init.zeros_(self.model[-1].bias)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.001)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.x=torch.empty((0,feature_dim),device=self.device);self.z=torch.empty(0,device=self.device);self.y=torch.empty(0,device=self.device)
        self.total_steps=0

    def normalize(self,x):
        return F.layer_norm(torch.as_tensor(x,dtype=torch.float32,device=self.device),(self.feature_dim,))

    def add(self,features,source_logits,teacher_probability):
        x=self.normalize(features);z=torch.as_tensor(source_logits,dtype=torch.float32,device=self.device)
        y=torch.as_tensor(teacher_probability,dtype=torch.float32,device=self.device)
        self.x=torch.cat([self.x,x]);self.z=torch.cat([self.z,z]);self.y=torch.cat([self.y,y])

    def fit(self,min_steps=400,max_steps=4000,excess_tolerance=.005):
        begin=time.perf_counter();n=len(self.y)
        if not n:return dict(status='NO_TEACHER_EVIDENCE',steps=0,total_steps=self.total_steps,teacher_examples=0,wall_seconds=0.)
        entropy=float((-(self.y*torch.log(self.y.clamp_min(1e-8))+(1-self.y)*torch.log((1-self.y).clamp_min(1e-8)))).mean())
        loss_value=None;steps=0
        for step in range(max_steps):
            ids=torch.randperm(n,generator=self.generator,device=self.device)[:512]
            logits=self.z[ids]+self.model(self.x[ids]).squeeze(-1)
            loss=F.binary_cross_entropy_with_logits(logits,self.y[ids])
            self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step();steps+=1
            if steps>=min_steps and steps%100==0:
                with torch.no_grad():loss_value=float(F.binary_cross_entropy_with_logits(self.z+self.model(self.x).squeeze(-1),self.y))
                if loss_value-entropy<=excess_tolerance:break
        self.total_steps+=steps
        return dict(status='PREFIX_TEACHER_FIT',steps=steps,total_steps=self.total_steps,teacher_examples=n,
            full_teacher_CE=loss_value,teacher_entropy=entropy,teacher_KL=loss_value-entropy,
            wall_seconds=time.perf_counter()-begin,normalization='per-row LayerNorm, no target batch statistics',
            no_GT_selection=True,min_steps=min_steps,max_steps=max_steps)

    @torch.no_grad()
    def predict(self,features,source_logits):
        z=torch.as_tensor(source_logits,dtype=torch.float32,device=self.device)
        return torch.sigmoid(z+self.model(self.normalize(features)).squeeze(-1)).cpu().numpy()

    def state(self):
        return dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            x=self.x,z=self.z,y=self.y,rng=self.generator.get_state(),total_steps=self.total_steps)

    def restore(self,state):
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        self.x=state['x'].to(self.device);self.z=state['z'].to(self.device);self.y=state['y'].to(self.device)
        self.generator.set_state(state['rng'].cpu());self.total_steps=state['total_steps']
