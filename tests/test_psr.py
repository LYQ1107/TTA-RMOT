from collections import Counter
import numpy as np
import torch
from flex_uda.teacher.psr_sampler import choose_tracks,ROLES,temporal_kind
from flex_uda.teacher.psr_supervision import direct_evidence,expand_local
from flex_uda.losses.psr_objectives import psr_losses


def test_fixed_roles_explore_low_source_not_force_positive():
    import random
    chosen=choose_tracks([1,2,3],{1:3,2:.01,3:-5},Counter(),'PSR',['SOURCE_LOW_VISIBLE','SOURCE_UNCERTAIN'],random.Random(17))
    assert [r[0] for r in chosen]==[3,2]
    assert temporal_kind('dancing person with purple pants')=='DYNAMIC_OR_MIXED'


def fixture(p=.99):
    observations={t:{7:dict(bbox_xyxy=[0,0,10,20],raw_mot_line=f'{t},7,0,0,10,20,1,-1,-1,-1')} for t in (1,2,3)}
    rows=[]
    for t in (1,3):
        s=dict(sample_id=str(t),video_id='v',query_id='q',query_text='dancing person',track_id=7,frame_id=t,anchor_frame=2,window_id='window',bbox_xyxy=[0,0,10,20])
        rows.append(direct_evidence(s,dict(yes_logit=4,no_logit=0,yes_probability=p,top1_is_yes_no=True),'fixed','manifest'))
    return rows,observations,{(t,7):np.ones(8) for t in (1,2,3)}


def test_expansion_is_only_one_finite_weak_middle_frame_and_no_new_teacher_call():
    rows,obs,appearance=fixture();expanded,rejected,audit=expand_local(rows,obs,appearance)
    assert len(expanded)==1 and expanded[0]['frame_id']==2 and expanded[0]['supervision_source']=='EXPANDED_WEAK'
    assert expanded[0]['observed_teacher_frames']==[1,3] and audit['extra_teacher_calls']==0 and not rejected


def test_no_negative_or_unknown_propagation():
    rows,obs,appearance=fixture(.01)
    assert not expand_local(rows,obs,appearance)[0]
    rows[0]['valid']=False
    assert not expand_local(rows,obs,appearance)[0]


def test_dynamic_id_break_and_appearance_change_reject_expansion():
    rows,obs,appearance=fixture();appearance[2,7]=np.arange(8)
    assert not expand_local(rows,obs,appearance)[0]
    rows,obs,appearance=fixture();obs[2]={}
    assert not expand_local(rows,obs,appearance)[0]


def test_invalid_teacher_stays_unknown_not_negative():
    row=direct_evidence(dict(frame_id=1),None,'revision','manifest')
    assert row['state']=='UNKNOWN' and not row['valid'] and row['teacher_prob']==.5


def test_direct_and_expanded_means_do_not_scale_with_duplicate_batch_rows():
    x=torch.tensor([0.,1.],requires_grad=True);p=torch.tensor([.9,.8]);valid=torch.ones(2,dtype=torch.bool)
    a=psr_losses(x,p,valid,x,p,valid)
    b=psr_losses(x.repeat(3),p.repeat(3),valid.repeat(3),x.repeat(5),p.repeat(5),valid.repeat(5))
    torch.testing.assert_close(a['total'],b['total']);a['total'].backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_parent_weight_control_reuses_actual_probabilities_and_frames_only():
    from flex_uda.teacher.psr_dataset import parent_weight_rows
    rows,obs,appearance=fixture();weak=expand_local(rows,obs,appearance)[0]
    replay=parent_weight_rows(rows,weak)
    assert len(replay)==2 and {r['frame_id'] for r in replay}=={1,3}
    assert {r['original_direct_sample_id'] for r in replay}=={'1','3'}
    assert all(r['teacher_prob']==.99 and r['supervision_source']=='DIRECT_PARENT_WEIGHT_CONTROL_NOT_NEW_LABEL' for r in replay)
    assert len({r['sample_id'] for r in replay})==2
