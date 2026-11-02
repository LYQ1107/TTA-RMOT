import itertools
import torch
from flex_uda.presence.cardinality import BernoulliCardinalityHead,ARCHITECTURE
from flex_uda.presence.residual import head_from_checkpoint


def test_exact_enumeration_normalization_presence_and_expectation():
    margin=torch.tensor([-.7,.2,1.4,-2.],dtype=torch.float64)
    head=BernoulliCardinalityHead(.7,1.8)
    output=head(margin);p=output['candidate_probabilities']
    truth=torch.zeros(5,dtype=torch.float64)
    for labels in itertools.product([0,1],repeat=4):
        probability=torch.prod(torch.stack([p[i] if label else 1-p[i] for i,label in enumerate(labels)]))
        truth[sum(labels)]+=probability
    assert torch.allclose(output['joint'],truth,atol=1e-12)
    assert abs(float(output['joint'].sum())-1)<1e-12
    assert torch.allclose(output['presence'].sigmoid(),1-truth[0],atol=1e-12)
    assert torch.allclose((truth*torch.arange(5)).sum(),output['expected_count'],atol=1e-12)


def test_empty_padding_permutation_and_more_than_ten():
    head=BernoulliCardinalityHead()
    empty=head(torch.empty(0));assert empty['joint'].tolist()==[1.] and empty['expected_count']==0
    m=torch.tensor([[2.,-1.,999.],[-4.,.3,5.]])
    mask=torch.tensor([[False,False,True],[True,True,True]])
    a=head(m,mask);b=head(m[:,[2,0,1]],mask[:,[2,0,1]])
    assert torch.allclose(a['joint'],b['joint'],atol=1e-12)
    assert a['joint'][1,0]==1 and a['joint'][1,1:].sum()==0
    assert int(head(torch.full((25,),2.))['logits'].argmax())>10


def test_monotonic_candidate_calibration_does_not_force_native_count():
    head=BernoulliCardinalityHead(.05,20.)
    values=torch.tensor([-10.,-1.,-.01,0.,.01,1.,10.])
    scaled=head.candidate_logits(values)
    assert (scaled.diff()>0).all() and torch.equal(values>0,scaled>0)
    # Both native margins are positive, but marginal cardinality mode is 1.
    out=BernoulliCardinalityHead()(torch.tensor([.1,.1]))
    assert int(out['logits'].argmax())==1


def test_strict_checkpoint_and_finite_probability_gradient():
    head=BernoulliCardinalityHead(.9,1.2)
    out=head(torch.tensor([-.3,.6]));loss=-out['joint'][1].log();loss.backward()
    assert torch.isfinite(head.slopes.grad).all() and head.slopes.grad.abs().sum()>0
    restored=head_from_checkpoint(dict(architecture=ARCHITECTURE,head=head.state_dict()))
    assert torch.equal(restored(torch.tensor([-.3,.6]))['joint'],out['joint'])
