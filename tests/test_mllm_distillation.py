import pytest
import torch
from flex_uda.losses.distillation import soft_distillation
from flex_uda.teacher.mllm_pseudo_dataset import digest,identity


def test_soft_BCE_matches_independent_binary_probability_and_gradient():
    margin=torch.tensor([2.,-1.,0.],requires_grad=True)
    target=torch.tensor([.8,.2,.6],requires_grad=True)
    loss=soft_distillation(margin,target,torch.tensor([True,True,False]))
    expected=torch.nn.functional.binary_cross_entropy_with_logits(margin[:2],target[:2].detach())
    torch.testing.assert_close(loss,expected);loss.backward()
    torch.testing.assert_close(margin.grad[:2],(margin.detach()[:2].sigmoid()-target.detach()[:2])/2)
    assert margin.grad[2]==0 and target.grad is None


def test_unlabeled_candidates_have_zero_loss_and_no_default_NEG():
    margin=torch.tensor([4.,-3.],requires_grad=True)
    loss=soft_distillation(margin,torch.tensor([.7,.5]),torch.tensor([False,False]))
    loss.backward();assert loss==0 and torch.count_nonzero(margin.grad)==0


def test_multiple_positive_people_do_not_compete_in_softmax():
    margin=torch.zeros(2,requires_grad=True)
    soft_distillation(margin,torch.ones(2),torch.ones(2,dtype=torch.bool)).backward()
    assert (margin.grad<0).all()


@pytest.mark.parametrize('probability',[float('nan'),-.1,1.01])
def test_invalid_probability_is_not_repaired_into_negative(probability):
    with pytest.raises(ValueError):
        soft_distillation(torch.zeros(1),torch.tensor([probability]),torch.tensor([True]))


def test_cache_identity_includes_video_query_frame_and_track():
    item=dict(model_revision='fixed',prompt_hash='p',image_sha256='i',bbox_hash='b',video_id='v1',query_id='q',frame_id=3,track_id=7)
    for key,value in [('video_id','v2'),('query_id','q2'),('frame_id',4),('track_id',8),('bbox_hash','b2')]:
        assert digest(item)!=digest(dict(item,**{key:value}))
    assert identity(item)==('v1','q',3,7)


def test_offline_calibration_uses_fixed_bins_without_fitting():
    from scripts.analyze_mllm_distillation import calibration
    result=calibration([0.,1.,.5],[False,True,True])
    assert result['Brier']==pytest.approx(1/12)
    assert result['ECE10']==pytest.approx(1/6)
    assert sum(r['n'] for r in result['bins'])==3
