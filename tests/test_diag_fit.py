import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from flex_uda.model.flexhook_uda import attach_adapter,adapter_kwargs,forward_uda
from flex_uda.losses.distillation import soft_distillation
from flex_uda.diag_fit import teacher_weight,metrics,fit_subset,target_windows,optimizer_groups


class Model(nn.Module):
    def __init__(self,dim=8):
        super().__init__()
        self.output_ca_qkv=nn.ModuleList([nn.Module() for _ in range(4)])
        for block in self.output_ca_qkv:
            block.attention=nn.Module()
            for name in ('wq','wk','wv','wo'):setattr(block.attention,name,nn.Linear(dim,dim,bias=False))
        self.head=nn.ModuleList([nn.Sequential(nn.Linear(dim,dim,bias=False),nn.GELU(),nn.Linear(dim,2,bias=False)) for _ in range(4)])
        for p in self.parameters():p.requires_grad_(False)

    def decode(self,outputs,text,mask,positions,boxes):
        x=outputs[0];scores=[]
        for block,head in zip(self.output_ca_qkv,self.head):
            x=sum(getattr(block.attention,n)(x) for n in ('wq','wk','wv','wo'))/4
            scores.append(head(x))
        return torch.stack(scores,dim=1),torch.tensor(0.)


def prepared(dim=8):
    return dict(outputs=[torch.randn(3,1,dim)],text=None,mask=None,positions=None,boxes=None,track_ids=[1,2,3])


@pytest.mark.parametrize('alpha,capacity',[(1.,False),(8.,False),(32.,False),(32.,True)])
def test_zero_B_keeps_forward_bitwise(alpha,capacity):
    model=Model();p=prepared();before=forward_uda(model,p,return_features=False)['all_logits'].clone()
    attach_adapter(model,'cpu',alpha=alpha,**(dict(rank=16,layers='all',projections=('wq','wk','wv','wo'),unfreeze=('head',)) if capacity else {}))
    assert torch.equal(before,forward_uda(model,p,return_features=False)['all_logits'])


def test_soft_BCE_default_bitwise_and_weighted_formula():
    torch.manual_seed(17);margin=torch.randn(20);p=torch.rand(20);valid=torch.rand(20)>.3
    before=torch.nn.functional.binary_cross_entropy_with_logits(margin[valid].float(),p[valid].float())
    assert torch.equal(before,soft_distillation(margin,p,valid))
    weight=3.7
    expected=-(weight*p[valid]*torch.nn.functional.logsigmoid(margin[valid])+(1-p[valid])*torch.nn.functional.logsigmoid(-margin[valid])).mean()
    torch.testing.assert_close(soft_distillation(margin,p,valid,weight),expected,atol=1e-6,rtol=0)


def test_default_and_full_capacity_counts():
    model=Model(768);attach_adapter(model,'cpu',**adapter_kwargs({'adapter':{'rank':4,'layers':[-2,-1]}}))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==24576
    model=Model(768);attach_adapter(model,'cpu',rank=16,layers='all',projections=('wq','wk','wv','wo'),alpha=32,unfreeze=('head',))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2758656
    with pytest.raises(ValueError):attach_adapter(Model(),'cpu',projections=('not_a_projection',))


@pytest.mark.parametrize('capacity',[False,True])
def test_strict_load_selected_legacy_and_capacity(tmp_path,monkeypatch,capacity):
    import flex_uda.evaluation as evaluate
    import scripts.run_flexhook_dance_pcd_tta as loader
    config=dict(adapter=dict(rank=16,layers='all',projections=['wq','wk','wv','wo'],alpha=32,unfreeze=['head']),protocol={}) if capacity else dict(protocol={})
    model=Model();attach_adapter(model,'cpu',**adapter_kwargs(config));p=prepared()
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:param.add_(.01)
    before=forward_uda(model,p,return_features=False)['all_logits'].clone()
    checkpoint=tmp_path/'state.pt'
    torch.save(dict(student=model.state_dict(),config=config,step=5,best_source_loss=.2,source_checkpoint_sha256=evaluate.SOURCE_SHA,test_time_update=False),checkpoint)
    monkeypatch.setattr(loader,'_load_model',lambda *a,**kw:(None,Model(),None))
    monkeypatch.setattr(evaluate,'file_sha',lambda _:evaluate.SOURCE_SHA)
    tokenizer=SimpleNamespace(encode=lambda *a,**kw:[5970])
    monkeypatch.setattr(evaluate.RobertaTokenizerFast,'from_pretrained',lambda *a,**kw:tokenizer)
    restored,_,_=evaluate.load_selected(checkpoint,tmp_path,tmp_path,'cpu')
    assert torch.equal(before,forward_uda(restored,p,return_features=False)['all_logits'])


def test_K1_five_step_loss_parity_and_frozen_versions():
    torch.manual_seed(17);a=Model();p=prepared();attach_adapter(a,'cpu');b=copy.deepcopy(a)
    opts=[torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=5e-5) for m in (a,b)]
    versions={n:x._version for n,x in a.named_parameters() if not x.requires_grad}
    for step in range(5):
        losses=[]
        for model,opt in zip((a,b),opts):
            opt.zero_grad();m=forward_uda(model,p,return_features=False)['margin']
            loss=soft_distillation(m,torch.tensor([.99,.1,.6]),torch.ones(3,dtype=torch.bool));losses.append(loss.detach().clone())
            (loss/1).backward();torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.);opt.step()
        assert torch.equal(*losses)
    assert all(x._version==versions[n] for n,x in a.named_parameters() if not x.requires_grad)


def test_GT_free_weight_and_identical_subsets(tmp_path):
    from flex_uda.training import protect_target_annotations
    attempts=protect_target_annotations()
    result=teacher_weight([dict(valid=True,teacher_prob=p) for p in (.99,.96,.4,.05)],dict(pos_weight='auto',pos_weight_cap=10))
    assert result['n_pos']==2 and result['n_neg']==2 and result['pos_weight']==1 and not attempts
    pseudo=SimpleNamespace(groups=[dict(window_id=str(i)) for i in range(2125)])
    hashes=[]
    for i in range(3):
        _,sha=fit_subset(pseudo,tmp_path/str(i));hashes.append(sha)
    assert len(set(hashes))==1
    assert [g['window_id'] for g in target_windows(pseudo,2,4)]==['8','9','10','11']


def test_pair_agreement_is_within_same_video_query_frame_only():
    rows=[dict(video_id='v',query_id='q',frame_id=1,valid=True,teacher_probability=p,student_margin=m) for p,m in ((.99,2),(.1,-1),(.2,3))]
    rows.append(dict(rows[0],video_id='other',student_margin=-100))
    r=metrics(rows)
    assert r['pair_agreement_pairs']==2 and r['pair_agreement']==.5


def test_generated_JSON_launch_config_keeps_small_head_LR_numeric(tmp_path):
    import json
    from scripts.train_psr import load_config
    path=tmp_path/'launch.yaml';path.write_text(json.dumps(dict(optimizer=dict(lr=.0002,head_lr=.00005))))
    value=load_config(path)
    assert value['optimizer']['head_lr']==.00005 and isinstance(value['optimizer']['head_lr'],float)


def test_DFIT_keeps_original_AMP_same_batch_backoff_before_numeric_failure():
    from flex_uda.diag_fit import inherited_amp_backoff
    class Scaler:
        scale=65536.
        def is_enabled(self):return True
        def get_scale(self):return self.scale
        def update(self,new_scale):self.scale=new_scale
    scaler=Scaler();retry=0
    for _ in range(8):retry,old=inherited_amp_backoff(scaler,retry)
    assert retry==8 and scaler.get_scale()==256.
    with pytest.raises(FloatingPointError):inherited_amp_backoff(scaler,retry)


def test_capacity_only_LoRA_and_head_mutate():
    model=Model();attach_adapter(model,'cpu',rank=16,layers='all',projections=('wq','wk','wv','wo'),alpha=32,unfreeze=('head',))
    before={n:p.detach().clone() for n,p in model.named_parameters()};versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(optimizer_groups(model,params,dict(lr=.0002,head_lr=.00005)))
    for _ in range(2):
        optimizer.zero_grad();forward_uda(model,prepared(),return_features=False)['margin'].sum().backward();optimizer.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    assert changed and any(n.startswith('head.') for n in changed) and all('lora_' in n or n.startswith('head.') for n in changed)
    assert all(p._version==versions[n] for n,p in model.named_parameters() if not p.requires_grad)
