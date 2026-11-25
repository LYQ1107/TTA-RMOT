"""Check the candidate's claimed algebraic/optimization properties."""
import copy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit
from flex_uda.active.relational_transport import fit,center_groups,transport,evidence_rows,horizontal_direction


def observations():
    rng=np.random.default_rng(11);rows=[]
    for group in range(9):
        x=rng.uniform(.05,.95,7);noise=rng.normal(0,.05,7)
        for i,(z,e) in enumerate(zip(x,noise)):
            rows.append(dict(group=('video',str(group),group,'FRAME_BOX'),x=float(z),
                teacher_logit=float(2.7*z+e+group/3),weight=float(1+i/7)))
    return rows


def test_within_cancels_fixed_nuisance_at_fixed_weights():
    original=observations();shifted=copy.deepcopy(original)
    for row in shifted:row['teacher_logit']+=23*(row['group'][2]-4)
    first,_=fit(original);second,_=fit(shifted)
    assert abs(first-second)<1e-9


def test_root_optimizer_matches_independent_scalar_minimization():
    rows=observations();rows[5]['teacher_logit']+=14
    beta,receipt=fit(rows);x,y,w,_=center_groups(rows)
    def objective(b):
        r=np.abs(b*x-y)
        return .005*b*b+np.sum(w*np.where(r<=1,.5*r*r,r-.5))
    reference=minimize_scalar(objective,bounds=(0,20),method='bounded',options={'xatol':1e-10})
    assert abs(beta-reference.x)<1e-5 and receipt['KKT_residual']<1e-8


def test_unexcited_or_opposite_evidence_does_not_invent_positive_slope():
    rows=observations()
    for row in rows:row['teacher_logit']=-3*row['x']
    assert fit(rows)[0]==0
    for row in rows:row['x']=.5
    beta,receipt=fit(rows)
    assert beta==0 and receipt['excited_groups']==0


def test_transport_bound_monotonicity_and_exact_fallback():
    base=np.full(11,.4);x=np.linspace(0,1,11)
    assert np.array_equal(transport(base,x,.5,0),base)
    assert np.array_equal(transport(base,x,.5,7,eligible=np.zeros(11,bool)),base)
    result=transport(base,x,.5,7)
    assert np.all(np.diff(result)>=0)
    assert np.max(np.abs(logit(result)-logit(base)))<=np.log(3)+1e-12


def test_exact_request_answer_multiplicity_and_pixel_scale():
    query=[dict(expression_id='cars-left',query_text='cars on the left')]
    first=dict(query_id='cars-left',track_id=1,frame_id=0,effective_view='FRAME_BOX',
        input=dict(bbox=[10,10,30,40],frame=0,query='cars-left'),yes_probability=.8)
    second=copy.deepcopy(first);second['input']['bbox']=[40,10,60,40];second['input']['frame']=1;second['frame_id']=1;second['yes_probability']=.2
    a,_=evidence_rows([first,second],query,100,'v')
    b,_=evidence_rows([first,second,first,second,second],query,100,'v')
    assert a==b and fit(a)[0]==fit(b)[0]
    enlarged=copy.deepcopy(first);enlarged['input']['bbox']=[2*x for x in first['input']['bbox']]
    c,_=evidence_rows([enlarged],query,200,'v')
    assert c[0]['x']==a[0]['x']
    assert horizontal_direction('white cars on the left')==-1
    assert horizontal_direction('cars to the left of the bus')==0
    assert horizontal_direction('left team player')==0
