"""Classical part-colour descriptors and query-aware visual cohort geometry."""
import re
import numpy as np
from flex_uda.active.acquisition import expression_types,COLORS

PARTS=((.1,.05,.9,.95),(.2,.1,.8,.55),(.2,.55,.8,.95))
PART_DIM=104


def part_histograms(image,box):
    x,y,w,h=box;features=[]
    for left,top,right,bottom in PARTS:
        bounds=(max(0,int(x+left*w)),max(0,int(y+top*h)),
            min(image.width,int(np.ceil(x+right*w))),min(image.height,int(np.ceil(y+bottom*h))))
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            # Off-screen body regions carry no colour observation. A zero
            # mass descriptor is missing data, not a black-clothing vote.
            features.append(np.zeros(PART_DIM));continue
        hsv=np.asarray(image.crop(bounds).convert('HSV'),dtype=np.int32).reshape(-1,3)
        hs=np.bincount((hsv[:,0]*12//256)*8+hsv[:,1]*8//256,minlength=96).astype(float)
        value=np.bincount(hsv[:,2]*8//256,minlength=8).astype(float)
        features.append(np.r_[.5*hs/hs.sum(),.5*value/value.sum()])
    return np.asarray(features)


def part_mask(text):
    text=text.lower();stripped=re.sub(r'\b(?:dark|light)[ -]side\b','',text)
    if not re.search(r'\b('+COLORS+r'|dark|light)\b',text):return np.zeros(3)
    if any(k in expression_types(stripped) for k in ('MOTION','SPATIAL')):return np.zeros(3)
    upper=bool(re.search(r'\b(tshirt|t-shirt|shirt|jersey|top|sweater)\b',text))
    lower=bool(re.search(r'\b(pants|trousers|skirt|shorts)\b',text))
    if upper or lower:return np.asarray([0.,float(upper),float(lower)])
    if re.search(r'\b(?:dark|light)[ -]side\b',text):return np.asarray([0.,1.,0.])
    return np.asarray([1.,0.,0.])


def augmented_features(pcd,colours,texts):
    pcd=np.asarray(pcd,dtype=float);pcd=pcd-pcd.mean(2,keepdims=True)
    pcd/=np.maximum(np.linalg.norm(pcd,axis=2,keepdims=True),1e-8)
    colour=np.stack([(colours*part_mask(text)[None,:,None]).reshape(len(colours),-1) for text in texts])
    colour/=np.maximum(np.linalg.norm(colour,axis=2,keepdims=True),1e-8)
    return np.concatenate([pcd,colour],axis=2).astype('float32'),pcd,colour
