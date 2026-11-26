"""Image/query appearance cues and teacher-only monotone regularization.

Physical colour prototypes are fixed in HSV, not fitted to target annotations.
The monotone map sees only past/current teacher probabilities. Its bounded
odds correction is a regularizer, not an independent Bayesian observation.
"""
import hashlib
import re
import numpy as np
from scipy.special import expit,logit
from flex_uda.active.color_cohort import part_mask

NAMES=('black','white','gray','red','orange','yellow','green','blue','purple','pink','brown')
COLOURS=r'black|white|gray|grey|silver|red|orange|yellow|green|blue|purple|pink|brown|dark|light'
ODDS_CAP=float(np.log(3.))


def canonical_colour(name):
    return dict(grey='gray',silver='gray',dark='black',light='white').get(name,name)


def colour_prototypes():
    h=(np.arange(12)+.5)[:,None,None]/12
    s=(np.arange(8)+.5)[None,:,None]/8
    v=(np.arange(8)+.5)[None,None,:]/8
    saturated=np.clip((s-.15)/.35,0,1)
    visible=np.clip((v-.15)/.3,0,1)
    table=dict(black=np.broadcast_to(np.clip((.35-v)/.25,0,1),(12,8,8)),
        white=np.broadcast_to((1-saturated)*np.clip((v-.45)/.35,0,1),(12,8,8)),
        gray=np.broadcast_to((1-saturated)*np.maximum(0,1-np.abs(v-.45)/.35),(12,8,8)))
    centers=dict(red=(0.,.09),orange=(1/12,.06),yellow=(1/6,.07),green=(1/3,.12),
        blue=(2/3,.12),purple=(5/6,.09),pink=(11/12,.09),brown=(1/12,.09))
    for name,(center,width) in centers.items():
        distance=np.minimum(np.abs(h-center),1-np.abs(h-center))
        value=np.exp(-.5*(distance/width)**2)*saturated*visible
        if name=='brown':value=value*np.clip((.7-v)/.3,0,1)
        elif name=='orange':value=value*np.clip((v-.4)/.3,0,1)
        elif name=='pink':value=value*(1-.5*saturated)*np.clip((v-.4)/.3,0,1)
        table[name]=np.broadcast_to(value,(12,8,8))
    weights=np.stack([table[name] for name in NAMES])
    return weights/np.maximum(weights.sum(0,keepdims=True),1e-12)


PROTOTYPES=colour_prototypes()


def colour_probabilities(colours):
    mass=np.asarray(colours,dtype=float)**2
    hs=mass[:,:,:96].reshape(mass.shape[:2]+(12,8));value=mass[:,:,96:]
    valid=(hs.sum((2,3))>0)&(value.sum(2)>0)
    hs/=np.maximum(hs.sum((2,3),keepdims=True),1e-12)
    value/=np.maximum(value.sum(2,keepdims=True),1e-12)
    # The stored descriptors contain HS and V marginals. Their product is an
    # explicit approximation; no unavailable pixel joint distribution is used.
    probability=np.einsum('nphs,npv,chsv->npc',hs,value,PROTOTYPES,optimize=True)
    brightness=value@((np.arange(8)+.5)/8)
    return probability,brightness,valid


def appearance_spec(text):
    text=text.lower()
    side=re.search(r'\b(dark|light)[ -]side\b',text)
    if side:return dict(family=1,kind='relative_upper_brightness',light=side.group(1)=='light')
    mask=part_mask(text)
    if not mask.any():return None
    colours=re.findall(r'\b('+COLOURS+r')\b',text)
    if not colours:return None
    if len(colours)==1 and re.search(r'\bdressed\s+all\s+in\b',text):
        return dict(family=0,kind='named_colour',attributes=[(1,canonical_colour(colours[0])),(2,canonical_colour(colours[0]))])
    garments=re.finditer(r'\b('+COLOURS+r')\s+(t-?shirt|shirt|jersey|top|sweater|pants|trousers|skirt|shorts)\b',text)
    attributes=[]
    for match in garments:
        part=2 if match.group(2) in ('pants','trousers','skirt','shorts') else 1
        attributes.append((part,canonical_colour(match.group(1))))
    if attributes:return dict(family=0,kind='named_colour',attributes=attributes)
    if len(set(colours))==1:
        return dict(family=0,kind='named_colour',attributes=[(int(p),canonical_colour(colours[0])) for p in np.flatnonzero(mask)])
    return None


def appearance_cues(colours,texts,lengths):
    probability,brightness,valid=colour_probabilities(colours)
    cue=np.full((len(texts),len(colours)),.5,dtype='float32');available=np.zeros(cue.shape,bool)
    families=np.full(len(texts),-1,dtype=int);specs=[appearance_spec(t) for t in texts]
    ids=np.flatnonzero(valid[:,1]);rank=np.full(len(colours),.5)
    if len(ids):
        b=brightness[ids,1];weight=np.asarray(lengths,dtype=float)[ids]
        rank[ids]=((b[None,:]<b[:,None])*weight[None,:]+.5*(b[None,:]==b[:,None])*weight[None,:]).sum(1)/weight.sum()
    for q,spec in enumerate(specs):
        if spec is None:continue
        families[q]=spec['family']
        if spec['kind']=='relative_upper_brightness':
            cue[q]=rank if spec['light'] else 1-rank;available[q]=valid[:,1]
        else:
            parts=[part for part,_ in spec['attributes']]
            values=np.stack([probability[:,part,NAMES.index(name)] for part,name in spec['attributes']])
            cue[q]=np.exp(np.log(np.maximum(values,1e-12)).mean(0));available[q]=valid[:,parts].all(1)
    return cue,available,families,specs


def pava_fit(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<8 or len(np.unique(x))<2:return None
    order=np.argsort(x,kind='stable');x=x[order];y=y[order]
    unique,start,count=np.unique(x,return_index=True,return_counts=True)
    sums=np.add.reduceat(y,start);blocks=[]
    for i,(total,n) in enumerate(zip(sums,count)):
        blocks.append([i,i,float(total),int(n)])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            right=blocks.pop();left=blocks.pop();blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    fitted=np.zeros(len(unique))
    for first,last,total,n in blocks:fitted[first:last+1]=total/n
    return unique,fitted


def shuffled_cues(cue,available,video):
    result=cue.copy()
    for q in range(len(cue)):
        ids=np.flatnonzero(available[q]);seed=int(hashlib.sha256((video+':'+str(q)+':semantic-placebo').encode()).hexdigest()[:8],16)
        result[q,ids]=cue[q,np.random.default_rng(seed).permutation(ids)]
    return result


def crossfit_predictions(cue,available,families,answers,tracks,video,history,*,monotone=True):
    probability=np.full(cue.shape,.5);prior=np.full(cue.shape,.5);active=np.zeros(cue.shape,bool);receipts=[]
    folds=np.asarray([int(hashlib.sha256((video+':'+str(int(t))+':semantic-fold').encode()).hexdigest()[:8],16)%5 for t in tracks])
    known=np.isfinite(answers)&available
    for fold in sorted(set(folds)):
        for family in (0,1):
            target=available&(families[:,None]==family)&(folds[None,:]==fold)
            if not target.any():continue
            training=known&(families[:,None]==family)&(folds[None,:]!=fold)
            past=history[family];x=np.r_[past['x'],cue[training]];y=np.r_[past['y'],np.clip(answers[training],.02,.98)]
            mean=float(np.clip(y.mean(),.02,.98)) if len(y) else .5
            fitted=pava_fit(x,y)
            if fitted is None:continue
            if monotone:predicted=np.interp(cue[target],*fitted)
            else:
                bins=np.minimum((x*10).astype(int),9);sums=np.bincount(bins,weights=y,minlength=10);counts=np.bincount(bins,minlength=10)
                predicted=((sums+2*mean)/(counts+2))[np.minimum((cue[target]*10).astype(int),9)]
            probability[target]=np.clip(predicted,.02,.98);prior[target]=mean;active[target]=True
            receipts.append(dict(fold=int(fold),family=family,teacher_examples=len(y),unique_cues=len(np.unique(x)),
                monotone=monotone,fitted_levels=len(np.unique(fitted[1])),teacher_prior=mean))
    return probability,prior,active,receipts


def correct_belief(unary,probability,prior,active):
    result=np.array(unary,copy=True)
    evidence=np.clip(logit(probability[active])-logit(prior[active]),-ODDS_CAP,ODDS_CAP)
    result[active]=expit(logit(np.clip(unary[active],1e-6,1-1e-6))+evidence)
    return result


def append_history(history,cue,available,families,answers):
    for family in (0,1):
        selected=np.isfinite(answers)&available&(families[:,None]==family)
        history[family]['x']=np.r_[history[family]['x'],cue[selected]]
        history[family]['y']=np.r_[history[family]['y'],np.clip(answers[selected],.02,.98)]
