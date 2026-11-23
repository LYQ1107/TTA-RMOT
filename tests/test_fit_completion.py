"""A real soft-label fit checks the additional baseline optimizer contract."""
import numpy as np
import torch
from torch import nn
from flex_uda.active.fit_completion import polish_head


def test_polish_reduces_weighted_teacher_loss_without_changing_capacity():
    torch.set_num_threads(1);torch.manual_seed(4)
    x=np.random.default_rng(3).normal(size=(40,5)).astype('float32')
    y=1/(1+np.exp(-(x[:,0]-.4*x[:,1])));weight=np.linspace(1.,3.,len(y))
    model=nn.Sequential(nn.Linear(5,64),nn.GELU(),nn.Linear(64,1))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    checkpoint=dict(model=model.state_dict(),mean=np.zeros(5),std=np.ones(5))
    result,receipt=polish_head(x,y,weight,checkpoint,device='cpu')
    assert receipt['teacher_KL']<receipt['initial_teacher_KL']
    assert receipt['teacher_fit_within_0_005']
    assert tuple(result['model']['0.weight'].shape)==(64,5)
    assert tuple(result['model']['2.weight'].shape)==(1,64)
