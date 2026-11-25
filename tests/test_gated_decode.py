import subprocess
import sys
import numpy as np
from scripts.gated_common import (global_moments,global_z,raw_decodes,gmm_reject,
    sampled_frames,gate_and_exclusivity,gated_masks,sa_mask,ROOT)

def test_global_query_not_video_standardization():
    meta=dict(episodes=[dict(query_text='a',start=0,stop=2),dict(query_text='a',start=2,stop=4),dict(query_text='b',start=4,stop=6)])
    a=np.array([0.,2.,10.,12.,7.,7.]);stats=global_moments(meta,a)
    assert stats['a']['mean']==6 and stats['a']['observations']==4
    assert stats['a']['std']==np.std(a[:4],ddof=0)
    z=global_z(np.array([[0.,2.],[7.,7.]]),[dict(query_text='a'),dict(query_text='b')],stats)
    assert np.all(z[0]<0) and np.all(z[1]==0)

def test_raw_lex_tie_and_causal_history():
    a=np.array([[3.,0.,0.],[1.,2.,4.]])
    x0,x1,mu=raw_decodes(a,np.zeros(3,dtype=int))
    assert x0[1].all() and x1[0,:2].all() and x1[1,2]
    b=a.copy();b[:,-1]=100
    assert np.array_equal(raw_decodes(b,np.zeros(3,dtype=int))[1][:,:2],x1[:,:2])

def test_sampling_half_up_nearest_lower_tie():
    assert sampled_frames(0,100,[4,6,18,30],[.05,.17,.29])==[4,18,30]
    assert sampled_frames(0,10,[0,1,2],[.05])==[1]

def test_gate_requires_distinct_frames_and_inclusive_probability():
    gate,r=gate_and_exclusivity({1:np.array([[.5,.1],[.6,.2]]),2:np.array([[.1,.8]])},2)
    assert not gate.any() and r['votes']==[2,1]
    gate,r=gate_and_exclusivity({1:np.array([[.5,.1]]),2:np.array([[.5,.1]])},2)
    assert gate.tolist()==[True,False]

def test_exclusivity_and_zero_denominator():
    _,r=gate_and_exclusivity({1:np.array([[.6,.7],[.8,.1],[.1,.2]])},2)
    assert r['E']==.5
    _,r=gate_and_exclusivity({1:np.array([[.1,.2]])},2)
    assert r['E'] is None and r['zero_denominator']

def test_structural_boundary_and_empty_user_rule():
    a=np.array([[2.,-1.],[1.,3.]]);i=np.array([0,1]);present=np.array([True,False])
    mask,mode=sa_mask(a,i,present,.5);assert mode=='TG_X0' and mask.sum()==1
    mask,mode=sa_mask(a,i,present,.5001);assert mode=='NATIVE' and np.array_equal(mask,a>0)
    mask,mode=sa_mask(a,i,present,None);assert not mask.any() and mode=='EMPTY_NO_TEACHER_EVIDENCE'

def test_gmm_exact_requested_fit_and_small_keep():
    mu=np.array([[-10.,-9.9,-10.1,5.,5.1,4.9]])
    mask,r=gmm_reject(mu,np.arange(6));assert r['BIC2']<r['BIC1'] and mask.sum()==3
    mask,r=gmm_reject(mu[:,:3],np.arange(3));assert not r['fit'] and mask.all()

def test_reassignment_can_be_empty_and_does_not_force_positive():
    a=np.array([[3.,1.],[1.,4.]]);i=np.array([0,1]);q=[dict(query_text='a'),dict(query_text='b')]
    stats={t:dict(mean=0.,std=1.) for t in ('a','b')}
    masks,_=gated_masks(a,i,np.array([False,True]),np.array([False,False]),stats,q)
    assert masks['TG_X0_REASSIGN'][1].all() and not masks['TG_EARLY_X1'].any()
    masks,_=gated_masks(a,i,np.array([False,False]),np.array([False,False]),stats,q)
    assert not masks['TG_X0_REASSIGN'].any()

def test_generation_annotation_audit_blocks_source_and_target():
    code='''from scripts.gated_common import deny_annotations
a=deny_annotations()
for p in ['/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance/labels.json','/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2/expression/0005/a.json']:
 try: open(p)
 except PermissionError: pass
 else: raise AssertionError(p)
assert len(a)==2
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
