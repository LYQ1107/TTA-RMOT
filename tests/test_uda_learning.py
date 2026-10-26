"""Necessary UDA semantics; real CUDA/checkpoint checks belong to GPU smoke."""
import copy
import torch
from torch import nn
from flex_tta.adaptation.pcd import LoRALinear
from flex_uda.model.flexhook_uda import ema_update
from flex_uda.losses.objectives import (supervised_ce,presence_probability,presence_loss,select_pseudo,pseudo_bce,
    consistency,semantic_kind,equivalent_query,exclusive_query_partner,exclusive_relation_loss)
from flex_uda.adaptation.tracklet_memory import TrackletMemory


def test_uda_binary_margin_is_binary_softmax_not_candidate_softmax():
    logits=torch.tensor([[1.,3.],[-2.,1.]])
    assert torch.allclose((logits[:,1]-logits[:,0]).sigmoid(),logits.softmax(-1)[:,1])


def test_multiple_positive_candidates_are_independent():
    margin=torch.tensor([4.,3.,-5.]);assert (margin>0).tolist()==[True,True,False]
    assert presence_probability(margin)>.99


def test_empty_prediction_is_allowed():
    assert (torch.tensor([-2.,-5.])>0).sum()==0


def test_unknown_mil_has_no_positive_or_negative_gradient():
    margin=torch.tensor([-2.,2.],requires_grad=True);loss=presence_loss(margin,'UNKNOWN');loss.backward()
    assert loss.item()==0 and torch.count_nonzero(margin.grad)==0


def test_negative_only_pseudo_tail_is_not_a_hard_training_batch():
    teacher=torch.tensor([-20.,-10.]);state,stats=select_pseudo(teacher,teacher,teacher)
    assert state.tolist()==[-1,-1] and stats['pseudo_negative']==0 and stats['proposed_negative']==2


def test_actual_confident_pos_and_neg_pseudo_are_retained_separately():
    value=torch.tensor([8.,-8.,0.]);state,stats=select_pseudo(value,value,value)
    assert state.tolist()==[1,0,-1] and stats['pseudo_positive']==stats['pseudo_negative']==1


def test_unknown_bce_is_skipped_not_relabelled_negative():
    margin=torch.tensor([1.,-1.],requires_grad=True);loss=pseudo_bce(margin,torch.tensor([-1,-1]));loss.backward()
    assert margin.grad.abs().sum()==0


def test_real_source_ce_backward_has_correct_shape_and_direction():
    logits=torch.zeros((2,4,1,2),requires_grad=True);loss=supervised_ce(logits,torch.tensor([1,0]));loss.backward()
    assert torch.isfinite(loss) and logits.grad[0,0,0,1]<0 and logits.grad[1,0,0,1]>0


def test_lora_zero_preserves_source_then_learns_without_changing_base():
    torch.manual_seed(17);base=nn.Linear(4,2,bias=False);module=LoRALinear(base,rank=2);x=torch.randn(3,4)
    original=base.weight.detach().clone();initial=base(x).detach()
    assert torch.equal(module(x),initial)
    optimizer=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=.01)
    nn.functional.cross_entropy(module(x),torch.tensor([1,0,1])).backward()
    assert module.lora_b.weight.grad.abs().sum()>0
    optimizer.step()
    assert not torch.equal(module(x),initial) and torch.equal(base.weight,original)


def test_ema_updates_only_authorized_trainable_parameters():
    student=nn.Linear(2,1);teacher=copy.deepcopy(student);teacher.weight.requires_grad_(False);teacher.bias.requires_grad_(False)
    student.bias.requires_grad_(False);before=teacher.weight.detach().clone();bias=teacher.bias.detach().clone()
    with torch.no_grad():student.weight.add_(1)
    count=ema_update(teacher,student,.9)
    assert count==2 and torch.allclose(teacher.weight,before+.1) and torch.equal(teacher.bias,bias)


def test_legal_equivalence_does_not_assume_query_is_present():
    assert equivalent_query('dancing person dressed all in black')=='a person who is dancing and dressed entirely in black'
    assert equivalent_query('dacning person with white Tshirt and purple pants') is None


def test_only_whole_outfit_colour_statements_are_mutually_exclusive():
    texts=['dancing person dressed all in black','dancing person dressed all in white','dancing person with white Tshirt and black pants']
    assert exclusive_query_partner(texts[0],texts)==texts[1]
    assert exclusive_query_partner(texts[2],texts) is None
    assert exclusive_relation_loss(torch.tensor([-5.]),torch.tensor([-5.]))==0


def test_logit_consistency_remains_finite_in_negative_tail():
    margin=torch.tensor([-20.,-10.],requires_grad=True);loss=consistency(margin,torch.tensor([-21.,-11.]));loss.backward()
    assert torch.isfinite(loss) and margin.grad.abs().sum()>0


def test_mixed_dynamic_query_is_not_frozen_by_identity_memory():
    query='dancing person dressed all in black';assert semantic_kind(query)=='MIXED'
    memory=TrackletMemory();memory.loss('v',query,1,[1],torch.tensor([2.]))
    loss,n=memory.loss('v',query,2,[1],torch.tensor([-2.],requires_grad=True))
    assert n==0 and loss==0


def test_static_consistency_requires_real_past_and_short_gap():
    memory=TrackletMemory();query='person wearing a black shirt'
    memory.loss('v',query,1,[1],torch.tensor([2.]))
    loss,n=memory.loss('v',query,2,[1],torch.tensor([-2.],requires_grad=True));assert n==1 and loss>0
    loss,n=memory.loss('v',query,100,[1],torch.tensor([2.]));assert n==0


def test_all_absent_presence_target_does_not_create_top1_positive():
    margin=torch.tensor([-2.,-3.],requires_grad=True);loss=presence_loss(margin,'SUPPORTED_ABSENT');loss.backward()
    assert (margin.grad>0).all()


def test_presence_is_explicit_fp32_under_autocast():
    margin=torch.tensor([-2.,1.],requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=presence_loss(margin,'SUPPORTED_PRESENT')
    loss.backward();assert loss.dtype==torch.float32 and torch.isfinite(margin.grad).all()
