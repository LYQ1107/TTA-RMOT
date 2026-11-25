"""Candidate relational transport from within-object teacher contrasts.

Classical ideas: eliminate stable nuisance effects by within-group centering,
fit a convex bounded-influence relation, and transport the relation to a new
observed state. This is a research candidate, not an accepted RMOT method.
No data files, manual labels, model weights or evaluation statistics are read.
"""
from collections import defaultdict
import hashlib
import json
import re
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logit
from flex_uda.active.acquisition import expression_types

MU=.01
HUBER_DELTA=1.
MAX_LOGIT_SHIFT=float(np.log(3.))


def horizontal_direction(text):
    low=text.lower();sides=set(re.findall(r'\b(left|right)\b',low))
    if len(sides)!=1 or 'MOTION' in expression_types(low):return 0
    # Team names and orientation/reference-object phrases do not establish
    # that absolute image coordinates are the intended reference frame.
    if re.search(r'\b(team|turn\w*|fac\w*|heading|direction|opposite|towards?|front|behind|beside|near|than)\b',low):return 0
    if re.search(r'\b(left|right)\s+(?:hand\s+|side\s+)?of\b|\bnext\s+to\b',low):return 0
    return 1 if next(iter(sides))=='right' else -1


def evidence_rows(records,queries,width,video):
    """Keep actual single-frame answers and distinguish correlated channels.

    Exact input+answer repetitions add no weight: unique probability values
    per exact request are reduced by a median, independent of multiplicity.
    Query/TrackID/view/video group intercepts remain separate.
    """
    texts={q['expression_id']:q['query_text'] for q in queries};unique={};eligible=0
    for row in records:
        direction=horizontal_direction(texts[row['query_id']])
        if not direction or row['effective_view'] not in ('CROP','FRAME_BOX'):continue
        eligible+=1;request=row['input']
        digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        box=request['bbox'];position=float(np.clip((float(box[0])+float(box[2]))/(2*width),0,1))
        value=dict(video=video,query=row['query_id'],track=int(row['track_id']),frame=int(row['frame_id']),
            view=row['effective_view'],x=position if direction>0 else 1-position,direction=direction,
            group=(video,row['query_id'],int(row['track_id']),row['effective_view']),request=digest)
        if digest not in unique:unique[digest]=dict(value,probabilities=set())
        unique[digest]['probabilities'].add(float(row['yes_probability']))
    result=[]
    for row in unique.values():
        row=dict(row);probabilities=row.pop('probabilities');p=float(np.median(sorted(probabilities)))
        assert np.isfinite(p) and 0<=p<=1
        row.update(probability=p,teacher_logit=float(logit(np.clip(p,.02,.98))),weight=1.+3.*abs(2*p-1))
        result.append(row)
    return result,dict(eligible_actual_question_records=eligible,unique_request_inputs=len(result),
        repeated_input_records=eligible-len(result),unique_response_median_not_independent_repeated_votes=True)


def center_groups(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row['group'])].append(row)
    design=[];response=[];weights=[];excited=0
    for values in grouped.values():
        x=np.asarray([r['x'] for r in values],float);y=np.asarray([r['teacher_logit'] for r in values],float)
        w=np.asarray([r['weight'] for r in values],float);w/=w.sum()
        dx=x-w@x;dy=y-w@y
        if float(w@(dx*dx))<=1e-12:continue
        design.extend(dx);response.extend(dy);weights.extend(w);excited+=1
    return np.asarray(design),np.asarray(response),np.asarray(weights),dict(
        total_groups=len(grouped),excited_groups=excited,centered_examples=len(design),
        no_excitation_is_no_identification=True,group_total_weight=1.)


def fit(rows,mu=MU,delta=HUBER_DELTA):
    assert mu>0 and delta>0
    x,y,w,counts=center_groups(rows)
    def gradient(beta):return float(mu*beta+np.sum(w*np.clip(beta*x-y,-delta,delta)*x))
    def objective(beta):
        residual=beta*x-y;a=np.abs(residual)
        return float(.5*mu*beta*beta+np.sum(w*np.where(a<=delta,.5*residual**2,delta*a-.5*delta**2)))
    if not len(x) or gradient(0)>=0:beta=0.
    else:
        upper=1.
        while gradient(upper)<0:upper*=2.
        beta=float(brentq(gradient,0.,upper,xtol=1e-11))
    derivative=gradient(beta);kkt=abs(derivative) if beta>0 else max(0.,-derivative)
    assert kkt<1e-8,(beta,derivative)
    return beta,dict(status='TEACHER_ONLY_WITHIN_OBJECT_RELATION_FIT',beta=beta,mu=mu,Huber_delta=delta,
        **counts,weighted_centered_Gram=float(np.sum(w*x*x)),gradient_at_solution=derivative,KKT_residual=kkt,
        objective_at_zero=objective(0.),objective_at_solution=objective(beta),
        beta_nonnegative=True,manual_target_labels_used=False,
        theory_scope='Strongly convex scalar fit, nuisance cancellation conditional on additive within-group logit model; optimization fit is not semantic accuracy or HOTA evidence.')


def transport(unary,position,reference,beta,eligible=None,cap=MAX_LOGIT_SHIFT):
    base=np.asarray(unary);position=np.asarray(position,float);reference=np.asarray(reference,float)
    assert beta>=0 and cap>0
    if beta==0:return base.copy()
    shift=np.clip(beta*(position-reference),-cap,cap)
    if eligible is not None:shift=np.where(eligible,shift,0.)
    result=base.astype(float,copy=True);active=np.broadcast_to(shift!=0,result.shape)
    changed=expit(logit(np.clip(base,1e-7,1-1e-7))+shift)
    result[active]=changed[active]
    return result
