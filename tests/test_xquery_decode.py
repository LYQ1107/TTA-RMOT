import numpy as np
from scripts.xquery_common import decode,causal_standardize,standardize,exact_f1_threshold

def test_x0_exact_original_track_argmax():
    from scripts.count_structure_common import track_mean
    rng=np.random.default_rng(0);x=rng.normal(size=(17,90));ids=np.tile(np.arange(3),30);f=np.repeat(np.arange(30),3)
    a=decode(x,f,ids,0.)
    old=np.stack([track_mean(v,ids,3) for v in x]).argmax(0)[ids]
    assert np.array_equal(a['X0_C3'],np.arange(17)[:,None]==old)
    assert np.all(a['X0_C3'].sum(0)==1)

def test_causal_decoder_prefix_invariance():
    rng=np.random.default_rng(1);x=rng.normal(size=(4,180));frames=np.repeat(np.arange(60),3);ids=np.tile(np.arange(3),60)
    a=decode(x,frames,ids,.2);b=x.copy();b[:,90:]+=1000;b=decode(b,frames,ids,.2)
    prefix=decode(x[:,:90],frames[:90],ids[:90],.2)
    for k in ('X1_C3_CAUSAL','X3_C3_Z_CAUSAL','X5_C3_Z_REJECT_CAUSAL'):
        assert np.array_equal(a[k][:,:90],b[k][:,:90])
        assert np.array_equal(a[k][:,:90],prefix[k])

def test_causal_moments_current_frame_inclusive_and_50_fallback():
    x=np.vstack([np.arange(60.),np.ones(60)]);f=np.repeat(np.arange(20),3);z=causal_standardize(x,f)
    assert np.array_equal(z[:,:48],x[:,:48])
    for a in (48,51,54,57):
        h=x[:,:a+3];sigma=h.std(1,ddof=0)
        expected=np.divide(x[:,a:a+3]-h.mean(1)[:,None],sigma[:,None],out=np.zeros((2,3)),where=sigma[:,None]!=0)
        assert np.allclose(z[:,a:a+3],expected,atol=1e-12)
    assert np.all(standardize(x)[1]==0)

def test_rejection_inclusive_ties_no_forced_positive():
    x=np.ones((3,60));f=np.repeat(np.arange(20),3);ids=np.tile(np.arange(3),20)
    yes=decode(x,f,ids,1.);no=decode(x,f,ids,np.nextafter(1.,np.inf))
    for k in ('X0_C3','X1_C3_CAUSAL','X2_C3_Z','X3_C3_Z_CAUSAL'):assert yes[k][0].all() and not yes[k][1:].any()
    for k in ('X4_C3_Z_REJECT','X5_C3_Z_REJECT_CAUSAL'):assert yes[k][0].all() and not no[k].any()

def test_exact_f1_search_and_smallest_tie():
    x=np.array([-2.,-2.,0.,1.,1.,2.]);y=np.array([0,1,0,1,1,0])
    fit=exact_f1_threshold(x,y);thresholds=np.r_[np.unique(x),np.nextafter(x.max(),np.inf)]
    f1=[]
    for t in thresholds:
        sel=x>=t;f1.append(2*(sel&y.astype(bool)).sum()/max(int(sel.sum()+y.sum()),1))
    assert fit['threshold']==thresholds[np.argmax(f1)] and fit['F1']==max(f1)
    assert exact_f1_threshold(x,np.zeros(len(x),dtype=int))['threshold']==-2.

def test_teacher_argmax_none_and_equal_half_retention():
    from scripts.xquery_teacher_combo import combo_masks
    a=np.full((17,3),.1);a[0,0]=.5;a[1,1]=.7
    b=np.zeros((18,3));b[17,0]=1.;b[2,1]=1.;b[3,2]=1.
    out=combo_masks(a,b)
    assert out['T1_ARGMAX'].sum()==3 and out['T1_ARGMAX'][0,2]
    assert out['T1_ARGMAX_REJECT'].sum()==2 and out['T1_ARGMAX_REJECT'][0,0]
    assert out['T2_ARGMAX'].sum()==2 and not out['T2_ARGMAX'][:,0].any()
