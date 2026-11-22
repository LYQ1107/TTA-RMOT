"""Teacher-only semantic unary, classical feature metrics and positive trust."""
import numpy as np
from flex_uda.active.color_cohort import augmented_features
from flex_uda.active.context_cohort import context_part_mask
from flex_uda.active.visual_classical import refine_distilled_prior


def query_geometry(pcd,colours,texts,kind='combined',hidden=None):
    _,normal,_=augmented_features(pcd,colours,texts)
    masks=np.stack([context_part_mask(t) for t in texts])
    rgb=np.stack([(colours*m[None,:,None]).reshape(len(colours),-1) for m in masks])
    rgb/=np.maximum(np.linalg.norm(rgb,axis=2,keepdims=True),1e-8)
    combined=np.concatenate([normal,rgb],axis=2).astype('float32')
    if kind=='combined':geometry=combined
    elif kind=='pcd':geometry=combined[:,:,:768]
    elif kind=='colour':geometry=combined[:,:,768:]
    elif kind=='hidden':geometry=np.asarray(hidden,dtype='float32')
    else:raise ValueError(kind)
    return masks,geometry


def positive_agreement_precision(early,final,protocol):
    """One fixed regularizer, not a calibrated probability or confidence CI.

    In a promised-presence task, two agreeing positive checkpoints may resist
    negative evidence propagated from other people. A queried node still has
    zero KD precision inside refine_distilled_prior, avoiding self-confirmation.
    CHECKLIST keeps the original rule and never assumes a target exists.
    """
    lower=np.minimum(early,final)
    if protocol!='PRESUPPOSED':return np.full(lower.shape,.1,dtype=float)
    agreement=np.maximum(0.,2.*lower-1.)
    return .1+3.9*agreement


def refine_queries(unary,pcd,colours,texts,teacher,*,kind='combined',hidden=None,
        graph=True,positive_guard=False,early=None,final=None,protocol=None):
    parts,geometry=query_geometry(pcd,colours,texts,kind,hidden)
    precision=positive_agreement_precision(early,final,protocol) if positive_guard else np.full(unary.shape,.1)
    result=np.array(unary,copy=True)
    for q,mask in enumerate(parts):
        if mask.any():
            result[q]=refine_distilled_prior(unary[q],geometry[q],teacher[q],graph=graph,
                source_weight=precision[q],node_mask=np.all(colours[:,mask>0].sum(2)>0,axis=1))
    return result
