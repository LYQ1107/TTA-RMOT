import torch
from flex_uda.presence.residual import ResidualCountHead, residual_loss, head_from_checkpoint, ARCHITECTURE


def inputs(margins):
    m = torch.tensor(margins, dtype=torch.float32)
    return dict(pcd=torch.randn(*m.shape, 768), margin=m, box=torch.rand(*m.shape, 4),
                track_id=torch.arange(m.shape[-1]).expand_as(m).long())


def test_native_initialization_all_empty_zero_ties_and_count_above_ten():
    model = ResidualCountHead().eval()
    rows = inputs([[3., 0., -1.], [-2., -3., -4.], [0., 0., 0.]])
    assert model(**rows)['logits'].argmax(-1).tolist() == [1, 0, 0]
    assert model(**inputs([[2.]*19]))['logits'].argmax(-1).item() == 19
    assert model(**inputs([[100., 1e-8, -1e-8]]))['logits'].argmax(-1).item() == 2
    empty = model(**inputs([[]]))
    assert empty['logits'].argmax(-1).item() == 0
    assert torch.isfinite(empty['presence']).all()


def test_padding_and_input_permutation_keep_count_distribution():
    model = ResidualCountHead().eval()
    torch.nn.init.normal_(model.residual[-1].weight, std=.02)
    row = inputs([[1., 1., -3., 10.]])
    row['track_id'] = torch.tensor([[9, 2, 5, 1]])
    row['padding_mask'] = torch.tensor([[False, False, False, True]])
    first = model(**row)
    perm = torch.tensor([2, 3, 0, 1])
    second = model(**{k: v[:, perm] for k, v in row.items()})
    assert torch.allclose(first['logits'], second['logits'], atol=1e-6)
    assert first['logits'].argmax(-1).item() <= 3


def test_count_supervision_finite_nonzero_gradient_and_strict_checkpoint():
    model = ResidualCountHead().eval()
    row = inputs([[2., 1., -1.], [1., -1., -2.]])
    output = model(**row)
    loss, ce, kl = residual_loss(output, torch.tensor([1, 0]))
    assert torch.isfinite(loss) and ce > 0 and abs(float(kl)) < 1e-6
    loss.backward()
    gradient = model.residual[-1].weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    restored = head_from_checkpoint(dict(architecture=ARCHITECTURE, head=model.state_dict())).eval()
    assert torch.equal(restored(**row)['logits'], output['logits'])


def test_track_ids_follow_candidate_dropout():
    from flex_uda.presence.data import collate_units
    row = {k: v[0] for k, v in inputs([[1.]*50]).items()}
    row.update(sent=torch.ones(768), y=torch.ones(50, dtype=torch.bool), count=50, index=0)
    batch = collate_units([row], augment=True)
    n = int((~batch['padding_mask'][0]).sum())
    assert batch['count'][0] == n == len(batch['track_id'][0])
