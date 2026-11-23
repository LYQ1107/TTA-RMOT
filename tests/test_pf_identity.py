import sys
from pathlib import Path
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flex_tta.adaptation.identity_contrast import (
    CausalIDMemory,IdentityContrastConfig,PCDFeatureTap,
)


def test_identity_contrast_autograd_without_query_pseudolabels():
    m=CausalIDMemory(IdentityContrastConfig(max_gap=2))
    m.observe(1,torch.tensor([[1.,0.],[0.,1.]]),[7,8])
    x=torch.tensor([[0.7,0.3],[0.3,0.7]],requires_grad=True)
    assert m.eligible(2,[7,8])
    loss,n=m.loss(2,x,[7,8]); assert n==2 and loss.item()>0
    loss.backward(); assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_identity_memory_rejects_stale_and_resets():
    m=CausalIDMemory()
    m.observe(1,torch.ones(2,3),[1,2])
    assert not m.eligible(10,[1,2])
    m.observe(10,torch.ones(2,3),[1,2])
    with pytest.raises(ValueError):m.observe(10,torch.ones(2,3),[1,2])
    m.reset();assert not m.eligible(1,[1,2])


def test_tap_shape_and_activation_is_differentiable():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_ca_qkv=nn.ModuleList([nn.Linear(3,3,bias=False)])
        def forward(self,x):return self.output_ca_qkv[0](x)
    model=Toy();tap=PCDFeatureTap(model)
    output=model(torch.ones(2,1,3,requires_grad=True))
    h=tap.take()
    assert h.shape==(2,3)
    h.pow(2).sum().backward()
    assert model.output_ca_qkv[0].weight.grad is not None
    with pytest.raises(RuntimeError):tap.take()
    tap.close()


def test_identity_current_and_future_features_are_ineligible():
    memory = CausalIDMemory()
    memory.observe(5, torch.eye(2), [1, 2])
    for frame in (4, 5):
        assert not memory.eligible(frame, [1, 2])
        _, support = memory.loss(frame, torch.eye(2), [1, 2])
        assert support == 0


def test_lora_episode_reset_restores_source_forward():
    from flex_tta.adaptation.pcd import PCDLoRATTA, trainable_parameters
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(3, 3, bias=False)
            self.wv = nn.Linear(3, 3, bias=False)
        def forward(self, x):
            return self.wq(x) + self.wv(x)
    model = nn.Module()
    blocks = [nn.Module() for _ in range(2)]
    for block in blocks:
        block.attention = Attention()
    model.output_ca_qkv = nn.ModuleList(blocks)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.ones(2, 3)
    def forward():
        return sum(block.attention(x) for block in model.output_ca_qkv)
    source = forward().detach().clone()
    adapter = PCDLoRATTA(model, rank=2)
    for _ in range(2):
        adapter.reset_episode()
        assert torch.equal(forward(), source)
        optimizer = torch.optim.AdamW(trainable_parameters(adapter), lr=0.01)
        optimizer.zero_grad()
        forward().square().sum().backward()
        optimizer.step()
        assert not torch.equal(forward(), source)
