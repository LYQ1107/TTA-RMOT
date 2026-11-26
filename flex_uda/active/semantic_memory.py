"""Classic HMM semantic memory over a fixed physical trajectory.

The None state is explicit for exclusive inventories. No target labels,
learned transition weights, or forced frame-level presence enter this module.
"""
import numpy as np
from scipy.special import expit,logit,logsumexp


def bounded_logit(probability):
    return np.clip(logit(np.clip(np.asarray(probability,dtype=float),1e-6,1-1e-6)),-6.,6.)


def blend_memory(frame, memory, *, reset=False, exclusive=False):
    frame=np.asarray(frame);memory=np.asarray(memory)
    # Caller expands the constant track memory along the observed time axis.
    logits=.5*(bounded_logit(frame)+bounded_logit(memory))
    if not reset:return expit(logits)
    if exclusive:
        best=frame.argmax(1);row=np.arange(len(frame));highest=frame[row,best]
        second=np.partition(frame,-2,axis=1)[:,-2] if frame.shape[1]>1 else np.zeros(len(frame))
        old=memory.argmax(1)
        clear=(highest>=.85)&(second<.5)&(memory[row,best]<=.15)
        clear &= (old!=best)|(memory.max(1)<.5)
        # A confident new class supersedes the old prototype. Temporal HMM
        # support is still needed; one contradictory frame is not a reset.
        logits[clear]=bounded_logit(frame[clear])
    else:
        clear=((frame>=.85)&(memory<=.15))|((frame<=.15)&(memory>=.85))
        logits=np.where(clear,bounded_logit(frame),logits)
    return expit(logits)


def categorical_smooth(probability,hazard=.02):
    """At most one referring class, plus None; exact forward/backward."""
    p=np.asarray(probability,dtype=float)
    emission=np.exp(np.column_stack([np.zeros(len(p)),bounded_logit(p)]))
    n,k=emission.shape
    if not n:return np.empty_like(p),np.zeros_like(p,dtype=bool)
    # The uniform-switch transition is a*I+b*11'. This is the exact same
    # forward/backward recursion in O(K), with normalization at every step.
    off=hazard/(k-1);diagonal=1-hazard-off
    forward=np.empty_like(emission);backward=np.ones_like(emission)
    forward[0]=emission[0]/emission[0].sum()
    for t in range(1,n):
        forward[t]=emission[t]*(diagonal*forward[t-1]+off*forward[t-1].sum())
        forward[t]/=forward[t].sum()
    for t in range(n-2,-1,-1):
        value=emission[t+1]*backward[t+1]
        backward[t]=diagonal*value+off*value.sum()
        backward[t]/=backward[t].sum()
    result=forward*backward;result/=result.sum(1,keepdims=True)
    choice=result.argmax(1)
    mask=np.arange(1,k)[None,:]==choice[:,None]
    return result[:,1:],mask


def binary_smooth(probability,hazard):
    """Vectorized independent binary HMMs for overlapping referring queries."""
    z=bounded_logit(probability);n,q=z.shape
    hazard=np.broadcast_to(np.asarray(hazard,dtype=float),(q,))
    stay=np.log1p(-hazard);switch=np.log(hazard)
    a0=np.empty_like(z);a1=np.empty_like(z);b0=np.zeros_like(z);b1=np.zeros_like(z)
    if not n:return z,np.zeros_like(z,dtype=bool)
    a0[0]=np.log(.5);a1[0]=z[0]+np.log(.5)
    for t in range(1,n):
        a0[t]=np.logaddexp(a0[t-1]+stay,a1[t-1]+switch)
        a1[t]=z[t]+np.logaddexp(a1[t-1]+stay,a0[t-1]+switch)
        scale=np.logaddexp(a0[t],a1[t]);a0[t]-=scale;a1[t]-=scale
    for t in range(n-2,-1,-1):
        b0[t]=np.logaddexp(stay+b0[t+1],switch+z[t+1]+b1[t+1])
        b1[t]=np.logaddexp(stay+z[t+1]+b1[t+1],switch+b0[t+1])
        scale=np.logaddexp(b0[t],b1[t]);b0[t]-=scale;b1[t]-=scale
    logits=a1+b1-a0-b0;result=expit(logits)
    return result,result>.5


def ema(probability,alpha=.2):
    result=np.asarray(probability,dtype=float).copy()
    for t in range(1,len(result)):result[t]=alpha*result[t]+(1-alpha)*result[t-1]
    return result
