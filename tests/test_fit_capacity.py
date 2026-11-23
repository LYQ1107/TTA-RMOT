import torch
from torch import nn
from flex_uda.active.fit_capacity import widen


def test_widen_preserves_existing_function_and_enables_new_features():
    torch.manual_seed(4)
    old = nn.Sequential(nn.Linear(5, 4), nn.GELU(), nn.Linear(4, 1)).double()
    x = torch.randn(13, 5, dtype=torch.float64)
    new = widen(old, 9, 8)
    assert torch.allclose(new(x), old(x), atol=1e-14, rtol=1e-14)
    assert new[0].out_features == 9
    new(x).square().mean().backward()
    assert new[-1].weight.grad[:, 4:].abs().sum() > 0
