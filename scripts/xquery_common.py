"""Fixed cross-query decoders. No target GT, model update or teacher call."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import count_structure_common as cs
from scripts import diag_attrib_v2_common as v2
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

RUN=ROOT/'results/runs/xquery_decode_20261101';PUBLIC=ROOT/'results/xquery_decode'
MARGINS=('FROZEN','T2V_D1_s17','T2V_D1_s23','T2V_D1_s42','T2V_D3_s17','SAME_SET_D1')
DECODERS=('X0_C3','X1_C3_CAUSAL','X2_C3_Z','X3_C3_Z_CAUSAL','X4_C3_Z_REJECT','X5_C3_Z_REJECT_CAUSAL')
SOURCE=Path('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2')
SOURCE_SCORE=ROOT/'results/runs/official_refer_kitti_v2_2gpu/result.json'
SOURCE_SHA='d6fdc6dff6d6bece1961e3b888ba15a31a88bfddf8557f8a34fc0eae44de5e46'

def setup():
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);v2.EVAL=RUN/'evaluation'

def require_plan():
    r=json.loads((RUN/'prereg_publication.json').read_text())
    if not r['remote_verified'] or file_sha(PUBLIC/'PREREG.md')!=r['prereg_sha256']:raise ValueError('publish fixed protocol first')
    return r

def inputs(verify=True):
    meta,old,groups,_=v2.load_bank();bank=json.loads((v2.CACHE/'bank.json').read_text())
    controls=json.loads((cs.T2V/'controls_seal.json').read_text())
    paths={'FROZEN':v2.CACHE/'FROZEN.npy','SAME_SET_D1':v2.CACHE/'D1.npy',
           **{m:Path(controls['score_cache_paths'][m]) for m in MARGINS if m.startswith('T2V')}}
    expected={'FROZEN':bank['cache_files_sha256']['FROZEN.npy'],'SAME_SET_D1':bank['cache_files_sha256']['D1.npy'],
              **{m:controls['score_cache_sha256'][m] for m in MARGINS if m.startswith('T2V')}}
    if verify:
        original=json.loads((ROOT/'results/runs/diag_attrib_20261031_v1/protected_inputs.json').read_text())
        checks=[dict(path=r['margins_path'],sha256=r['margins_sha256']) for r in original if r['model'] in ('FROZEN','D1')]
        native=json.loads((cs.T2V/'predictions_seal.json').read_text())
        checks.extend(dict(path=e['margins_path'],sha256=e['margins_sha256']) for m,r in native['methods'].items() for e in r['episodes'])
        for r in checks:
            if file_sha(r['path'])!=r['sha256']:raise ValueError('changed sealed original margin '+r['path'])
        for m,p in paths.items():
            if file_sha(p)!=expected[m]:raise ValueError('changed sealed array '+m)
        atomic_json(RUN/'input_audit.json',dict(status='PASS',original_margin_files=len(checks),files=checks,
            arrays={m:dict(path=str(p),sha256=expected[m]) for m,p in paths.items()},candidate_query_rows=3588445,episodes=425))
    return meta,{m:np.load(p,mmap_mode='r') for m,p in paths.items()},groups

def standardize(matrix):
    mu=matrix.mean(axis=1,keepdims=True);sigma=matrix.std(axis=1,ddof=0,keepdims=True)
    return np.divide(matrix-mu,sigma,out=np.zeros_like(matrix,dtype=np.float64),where=sigma!=0)

def causal_standardize(matrix,frames):
    """Current-frame-inclusive moments; each past z keeps its own time origin."""
    out=np.empty_like(matrix,dtype=np.float64);mean=np.zeros(len(matrix));m2=mean.copy();count=0
    unique,starts=np.unique(frames,return_index=True)
    for a,b in zip(starts,np.r_[starts[1:],len(frames)]):
        block=matrix[:,a:b];n=b-a;bm=block.mean(axis=1);delta=bm-mean;total=count+n
        m2=m2+((block-bm[:,None])**2).sum(axis=1)+delta*delta*(count*n/total)
        mean=mean+delta*(n/total);count=total
        if count<50:out[:,a:b]=block
        else:
            sigma=np.sqrt(np.maximum(m2/count,0))
            out[:,a:b]=np.divide(block-mean[:,None],sigma[:,None],out=np.zeros_like(block),where=sigma[:,None]!=0)
    return out

def decode(matrix,frames,inverse,theta=None):
    """All six and only six rules, matrix rows already lexicographic queries."""
    matrix=np.asarray(matrix,dtype=np.float64);nq,n=matrix.shape;nt=int(inverse.max())+1
    mean=np.stack([cs.track_mean(x,inverse,nt) for x in matrix])
    causal=np.stack([cs.causal_track_mean(x,inverse,nt) for x in matrix])
    z=standardize(matrix);zc=causal_standardize(matrix,frames)
    zm=np.stack([cs.track_mean(x,inverse,nt) for x in z])
    zcm=np.stack([cs.causal_track_mean(x,inverse,nt) for x in zc])
    assignments=[mean.argmax(0)[inverse],causal.argmax(0),zm.argmax(0)[inverse],zcm.argmax(0)]
    out={d:np.arange(nq)[:,None]==a[None,:] for d,a in zip(DECODERS[:4],assignments)}
    if theta is not None:
        out[DECODERS[4]]=out[DECODERS[2]]&(mean[:,inverse]>=theta)
        out[DECODERS[5]]=out[DECODERS[3]]&(causal>=theta)
    return out

def exact_f1_threshold(values,labels):
    """Exact >= threshold search; equal rational F1 chooses smallest value."""
    values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    order=np.argsort(values,kind='stable');x=values[order];y=labels[order]
    unique,starts=np.unique(x,return_index=True);prefix=np.r_[0,np.cumsum(y)];positive=int(y.sum())
    best=None
    for threshold,start in zip(np.r_[unique,np.nextafter(unique[-1],np.inf)],np.r_[starts,len(x)]):
        tp=positive-int(prefix[start]);n=len(x)-int(start);den=n+positive
        num=2*tp
        if best is None or num*best['den']>best['num']*max(den,1):
            best=dict(threshold=float(threshold),num=num,den=max(den,1),TP=tp,FP=n-tp,FN=positive-tp,selected=n)
    best.update(F1=best['num']/best['den'],positive_tracks=positive,query_track_pairs=len(x),candidate_thresholds=len(unique)+1,
                retention='raw_mean >= theta; ties in maximum F1 choose smallest candidate threshold')
    return best

def checkpoints():
    sealed=json.loads((cs.T2V/'full_completion.json').read_text())['arms']
    result={m:dict(path=str(cs.T2V/'students'/m/'checkpoint_final_step.pt'),sha256=sealed[m]['checkpoint_sha256']) for m in MARGINS if m.startswith('T2V')}
    p=ROOT/'results/runs/diag_fit_20261030_v1/attempts/full/DIAG_FIT_OPT/initial/checkpoint_final_step.pt'
    meta=json.loads((ROOT/'results/runs/diag_fit_20261030_v1/predictions_seal.json').read_text())['methods']['DIAG_FIT_OPT']
    result['SAME_SET_D1']=dict(path=str(p),sha256=meta['checkpoint_sha256'])
    return result
