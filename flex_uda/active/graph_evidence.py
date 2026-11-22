"""Teacher-only comparison of visual, global and independent graph models.

The compatibility score integrates a free query intercept under a Gaussian
logit model. All kernels have the same marginal scale. Its softmax weights are
modeling assumptions, not calibrated probabilities that a teacher is correct.
No fitted KD output or target annotation enters model comparison.
"""
import numpy as np
from scipy.special import expit,logit,softmax
from flex_uda.active.visual_classical import visual_graph

NAMES=('VISUAL','GLOBAL','INDEPENDENT')


def graph_models(features,valid):
    n=len(features);ids=np.flatnonzero(valid);local,_=visual_graph(np.asarray(features)[ids])
    full=np.zeros((n,n));full[np.ix_(ids,ids)]=local
    uniform=np.zeros((n,n))
    if len(ids)>1:
        block=np.full_like(local,local.sum()/(len(ids)*(len(ids)-1)))
        np.fill_diagonal(block,0.);uniform[np.ix_(ids,ids)]=block
    weights=[full,uniform,np.zeros_like(full)];kernels=[]
    for matrix in weights:
        laplacian=np.diag(matrix.sum(1))-matrix
        covariance=np.linalg.inv(laplacian+.1*np.eye(n))
        scale=np.sqrt(np.diag(covariance));correlation=covariance/(scale[:,None]*scale[None,:])
        # Same marginal logit scale for all topologies; 4 is the existing
        # teacher-logit cap, not a target-GT fitted kernel amplitude.
        kernels.append(16.*correlation)
    return weights,kernels


def observation_system(kernel,answers):
    ids=np.flatnonzero(np.isfinite(answers));p=np.asarray(answers)[ids]
    target=logit(np.clip(p,.02,.98));precision=1+3*np.abs(2*p-1)
    system=kernel[np.ix_(ids,ids)]+np.diag(1/precision)
    return ids,target,system


def model_weights(kernels,answers):
    known=np.isfinite(answers)
    if known.sum()<2:
        return np.asarray([0.,0.,1.]),dict(log_scores=[0.,0.,0.],reason='fewer than two comparative teacher observations')
    scores=[]
    for kernel in kernels:
        ids,y,system=observation_system(kernel,answers);one=np.ones(len(ids))
        inverse_one=np.linalg.solve(system,one);inverse_y=np.linalg.solve(system,y)
        mean=float(one@inverse_y/(one@inverse_one));residual=y-mean
        sign,logdet=np.linalg.slogdet(system);assert sign>0
        # The common (k-1)*log(2*pi) cancels between the three candidates.
        score=-.5*(logdet+np.log(one@inverse_one)+residual@np.linalg.solve(system,residual))
        scores.append(float(score))
    return softmax(scores),dict(log_scores=scores,reason='same-scale restricted Gaussian evidence with free query intercept')


def teacher_prediction(kernel,answers,test_node):
    ids,y,system=observation_system(kernel,answers)
    if not len(ids):return .5
    one=np.ones(len(ids));inverse_one=np.linalg.solve(system,one);inverse_y=np.linalg.solve(system,y)
    mean=float(one@inverse_y/(one@inverse_one))
    conditional=mean+kernel[int(test_node),ids]@np.linalg.solve(system,y-mean)
    return float(expit(conditional))


def leave_one_out(kernels,answers):
    rows=[]
    for node in np.flatnonzero(np.isfinite(answers)):
        training=np.asarray(answers).copy();target=float(training[node]);training[node]=np.nan
        weights,record=model_weights(kernels,training)
        probability=np.asarray([teacher_prediction(k,training,node) for k in kernels])
        rows.append(dict(node=int(node),target=target,weights=weights.tolist(),
            VISUAL=float(probability[0]),GLOBAL=float(probability[1]),INDEPENDENT=float(probability[2]),
            BMA=float(weights@probability),MAP=float(probability[int(weights.argmax())]),EQUAL=float(probability.mean())))
    return rows
