"""Read-only sealed score banks and shared definitions; NO target GT IO."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import diag_attrib_v2_common as v2
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.diag_pf_countmatched import topk_ids

RUN=ROOT/'results/runs/count_structure_20261031'
PUBLIC=ROOT/'results/count_structure'
MARGINS=('FROZEN','T2V_D1_s17','T2V_D3_s17','SAME_SET_D1')
SELECTORS=('C1_TRACK_MEAN','C1c_TRACK_CAUSAL','C2_QUERY_ARGMAX','C3_QUERY_ARGMAX_TRACK',
           'C4_C3_SHIFT','C5_EM_LABEL_SHIFT','C6_EM_TRACK')
ORACLES=('V_NATIVE1','V_MEAN','F_NATIVE1','F_MEAN','TRACK_ORACLE','F_FRAME')
OLD=v2.RUN
T2V=ROOT/'results/runs/train2val_20261031'
EPS=1e-12

def setup():
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);v2.EVAL=RUN/'evaluation'

def require_plan():
    p=RUN/'prereg_publication.json'
    r=json.loads(p.read_text())
    if not r['remote_verified'] or file_sha(PUBLIC/'PREREG.md')!=r['prereg_sha256']:raise ValueError('publish fixed plan before execution')
    return r

def load_inputs(verify=True):
    meta,old,groups,_=v2.load_bank();bank=json.loads((OLD/'score_bank/bank.json').read_text())
    control=json.loads((T2V/'controls_seal.json').read_text())
    paths={'FROZEN':OLD/'score_bank/FROZEN.npy','SAME_SET_D1':OLD/'score_bank/D1.npy',
           **{m:Path(control['score_cache_paths'][m]) for m in MARGINS if m.startswith('T2V')}}
    expected={'FROZEN':bank['cache_files_sha256']['FROZEN.npy'],'SAME_SET_D1':bank['cache_files_sha256']['D1.npy'],
              **{m:control['score_cache_sha256'][m] for m in MARGINS if m.startswith('T2V')}}
    if verify:
        checks=[]
        protected=json.loads((ROOT/'results/runs/diag_attrib_20261031_v1/protected_inputs.json').read_text())
        for e in protected:
            if e['model'] not in ('FROZEN','D1'):continue
            if file_sha(e['margins_path'])!=e['margins_sha256']:raise ValueError('changed original margin')
            checks.append(dict(path=e['margins_path'],sha256=e['margins_sha256']))
        for m,r in json.loads((T2V/'predictions_seal.json').read_text())['methods'].items():
            if m not in MARGINS:continue
            for e in r['episodes']:
                if file_sha(e['margins_path'])!=e['margins_sha256']:raise ValueError('changed Train→Val margin')
                checks.append(dict(path=e['margins_path'],sha256=e['margins_sha256']))
        for m,p in paths.items():
            if file_sha(p)!=expected[m]:raise ValueError('changed sealed score array '+m)
        atomic_json(RUN/'input_audit.json',dict(status='PASS',margin_files=len(checks),files=checks,
            score_arrays={m:dict(path=str(p),sha256=expected[m]) for m,p in paths.items()},
            candidate_query_rows=3588445,episodes=425))
    return meta,{m:np.load(paths[m],mmap_mode='r') for m in MARGINS},groups

def video_banks(meta):
    by=defaultdict(list)
    for q in meta['episodes']:by[q['video_id']].append(q)
    for video,queries in sorted(by.items()):
        queries.sort(key=lambda q:q['expression_id']);tracks=v2.load_tracker_lines(v2.TRACKS,video)
        frame=np.concatenate([np.repeat(t,len(rows)) for t,rows in tracks.items()])
        ids=np.concatenate([np.asarray(sorted(rows),dtype=np.int64) for rows in tracks.values()])
        if len(queries)!=17 or any(q['stop']-q['start']!=len(ids) for q in queries):raise ValueError('unaligned17-query bank')
        unique,inverse=np.unique(ids,return_inverse=True)
        yield video,queries,tracks,frame,ids,unique,inverse

def track_mean(values,inverse,count):
    den=np.bincount(inverse,minlength=count)
    return np.bincount(inverse,weights=values,minlength=count)/den

def causal_track_mean(values,inverse,count):
    out=np.empty(len(values),dtype=np.float64)
    for j in range(count):
        idx=np.flatnonzero(inverse==j);out[idx]=np.cumsum(values[idx],dtype=np.float64)/np.arange(1,len(idx)+1)
    return out

def em_prior(values,a,b,pi_s,max_iter=100,tol=1e-6):
    base=a*np.asarray(values,dtype=np.float64)+b;pi=pi_s;converged=False
    for step in range(1,max_iter+1):
        post=expit(base+logit(np.clip(pi,EPS,1-EPS))-logit(pi_s));new=float(post.mean())
        if abs(new-pi)<tol:converged=True;pi=new;break
        pi=new
    posterior=expit(base+logit(np.clip(pi,EPS,1-EPS))-logit(pi_s))
    return posterior,dict(pi=pi,iterations=step,converged=converged,posterior_mean=float(posterior.mean()))

def masks_for_video(matrix,inverse,source_mean,global_mean,calibration):
    nquery,n=matrix.shape;count=int(inverse.max()+1);means=np.stack([track_mean(x,inverse,count) for x in matrix])
    best=matrix.argmax(axis=0);assignment=means.argmax(axis=0)
    out={s:np.zeros((nquery,n),dtype=bool) for s in SELECTORS};priors=[]
    for j,x in enumerate(matrix):
        out['C1_TRACK_MEAN'][j]=means[j,inverse]>0
        out['C1c_TRACK_CAUSAL'][j]=causal_track_mean(x,inverse,count)>0
        out['C2_QUERY_ARGMAX'][j]=best==j
        out['C3_QUERY_ARGMAX_TRACK'][j]=assignment[inverse]==j
        out['C4_C3_SHIFT'][j]=(assignment[inverse]==j)&(means[j,inverse]-global_mean+source_mean>0)
        posterior,detail=em_prior(x,calibration['a'],calibration['b'],calibration['pi_s']);priors.append(detail)
        out['C5_EM_LABEL_SHIFT'][j]=posterior>.5
        out['C6_EM_TRACK'][j]=track_mean(posterior,inverse,count)[inverse]>.5
    return out,priors

def frame_mask(q,ids,values,counts):
    """Exact v2 ranks with the original topk_ids numerical-ID tie rule."""
    out=np.zeros(len(values),dtype=bool)
    for t,a,b,g in q['groups']:
        a-=q['start'];b-=q['start'];k=min(b-a,max(0,int(counts[g])))
        if not k:continue
        scores=dict(zip(ids[a:b].tolist(),values[a:b].tolist()))
        chosen=topk_ids(ids[a:b].tolist(),scores,k)
        out[a:b]=np.isin(ids[a:b],list(chosen))
    return out

def granularity_verdict(video_mean,track_oracle,frame_mean):
    if video_mean>=31.771:return '视频级存在与数量即可达到同域官方水平'
    if video_mean<31.771<=track_oracle:return '需要轨迹级判断'
    if track_oracle<31.771<=frame_mean:return '需要帧级存在'
    return '需要帧级数量'
